"""The boundary between tensors and the CuPy arrays its kernels operate on.

These helpers move values between Tensor/Storage and device arrays. Numerical
helpers belong here only when several kernel families need them.
"""

from __future__ import annotations

import cupy
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def tensor_to_logical_array(tensor: Tensor) -> Any:
    """Return compact logical values as a device array.

    Kernels operate on compact arrays. Tensor metadata remains the source of
    truth, so a future non-compact layout is gathered before crossing this
    boundary rather than being reshaped as if logical and physical positions
    were identical.
    """
    storage = tensor._logical_storage_for("cuda")
    return storage.buffer.reshape(tensor.shape)


def _widen(values: Any) -> Any:
    """Return a binary64 array, keeping binary32 subnormals.

    ``astype`` flushes a binary32 subnormal on the way *up*: the smallest
    binary32 subnormal widens to zero, so a kernel that reads its operands
    into a binary64 working precision has lost them before any arithmetic
    runs. The PTX conversion does not. Section 5.4 requires gradual underflow
    and does not exempt a format crossing.
    """
    if values.dtype != cupy.float32:
        return cupy.asarray(values, dtype=cupy.float64)
    from tensors.backend.cuda.kernels.arithmetic import _ieee32

    return _ieee32.widen(values)


def _narrow(values: Any, target: Any) -> Any:
    """Round a binary64 array to ``target``, keeping binary32 subnormals."""
    if target != cupy.float32:
        return cupy.asarray(values, dtype=target)
    from tensors.backend.cuda.kernels.arithmetic import _ieee32

    return _ieee32.narrow(values)


def _working_values(tensor: Tensor) -> Any:
    """A tensor's values in binary64 working precision, subnormals kept.

    The unary elementwise kernels evaluate in binary64 and narrow once, so
    this is the first of the two format crossings; :func:`_storage` performs
    the second.
    """
    return _widen(tensor_to_logical_array(tensor))


def _errstate(**settings: str) -> Any:
    """Suppress the CuPy warnings a kernel handles through its own checks."""
    factory = getattr(cupy, "errstate", None)
    if factory is None:
        return nullcontext()
    return factory(**settings)


def _operand(value: Tensor | Scalar, dtype: DataType) -> Any:
    """Return a device operand with Python-reference working precision.

    Integer arithmetic has no exact device equivalent of the reference's
    arbitrary-precision Python integers, so integer dtypes are refused here and
    the caller declines in favour of the Python kernel.
    """
    from tensors.tensor import Tensor

    if dtype.kind == "integer":
        raise TypeError("CUDA integer kernels require the Python fallback")
    result = tensor_to_logical_array(value) if isinstance(value, Tensor) else value
    # The same crossing as _working_values, reached through a second helper.
    # cupy.asarray widens with astype, which flushes a binary32 subnormal.
    return _widen(cupy.asarray(result))


def _storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Retain a device result, or decline when it cannot be represented."""
    if dtype.kind == "integer":
        # Matches _operand: exact integer semantics stay on the Python kernel.
        return None
    flattened = cupy.asarray(result).reshape(-1)
    try:
        # Both crossings go through PTX; see _widen and _narrow. A result
        # already in binary64 is unchanged by the first.
        flattened = _widen(flattened)
    except (OverflowError, TypeError, ValueError):
        return None
    target_dtype = cupy.dtype(dtype.name)
    if target_dtype.itemsize < cupy.dtype(cupy.float64).itemsize:
        # A value the narrower dtype would round to infinity is not an overflow
        # the reference implementation produces; decline instead.
        finite = cupy.isfinite(flattened)
        outside_range = cupy.abs(flattened) > cupy.finfo(target_dtype).max
        if bool(cupy.any(finite & outside_range)):
            return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        contiguous = _narrow(flattened, target_dtype)
    storage = CudaStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage


def _finite_operands(*operands: Any) -> bool:
    """Return whether every operand is free of infinities and NaN."""
    finite = cupy.asarray(True)
    for operand in operands:
        finite = finite & cupy.all(cupy.isfinite(operand))
    return bool(finite)


def _shape_size(shape: tuple[int, ...]) -> int:
    return Shape.from_iterable(shape).size


# ----------------------------------------------------------------------
#  Elementwise arithmetic
#
#  docs/arithmetic-semantics.md governs +, -, * and /. Arithmetic works in
#  the declared dtype: integers wrap at their width and floating operands
#  keep their precision, so neither the object-array path nor the float64
#  working precision used by ``_operand`` applies here. These are separate
#  from the helpers above so that the operations outside that contract keep
#  the behaviour they were written against.
# ----------------------------------------------------------------------


def _arithmetic_storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Retain an arithmetic result at its declared dtype.

    Unlike ``_storage`` this never declines: an overflow to infinity and an
    integer that wrapped are both specified results, not reasons to hand the
    work to another backend.
    """
    native = cupy.dtype(dtype.name)
    flattened = cupy.asarray(result).reshape(-1)
    if flattened.dtype != native:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            flattened = flattened.astype(native, copy=False)
    storage = CudaStorage(cupy.ascontiguousarray(flattened), dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage
