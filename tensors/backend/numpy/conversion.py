"""The boundary between tensors and the NumPy arrays its kernels operate on.

These helpers move values between Tensor/Storage and native arrays. Numerical
helpers belong here only when several kernel families need them.
"""

from __future__ import annotations

import numpy
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _view(tensor: Tensor) -> Any:
    """Return compact logical values as a NumPy array.

    Kernels operate on compact arrays. Tensor metadata remains the source of
    truth, so a future non-compact layout is gathered before crossing this
    boundary rather than being reshaped as if logical and physical positions
    were identical.
    """
    storage = tensor._logical_storage_for("numpy")
    return storage.buffer.reshape(tensor.shape)


def _errstate(**settings: str) -> Any:
    """Suppress the NumPy warnings a kernel handles through its own checks."""
    factory = getattr(numpy, "errstate", None)
    if factory is None:
        return nullcontext()
    return factory(**settings)


def _operand(value: Tensor | Scalar, dtype: DataType) -> Any:
    """Return an array operand with Python-reference working precision.

    Integer results stay exact by computing in ``object`` arrays of Python
    integers rather than in a fixed-width integer type.
    """
    from tensors.tensor import Tensor

    result = _view(value) if isinstance(value, Tensor) else value
    working_dtype = numpy.float64 if dtype.kind == "floating" else object
    return numpy.asarray(result, dtype=working_dtype)


def _storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Retain a native array result, or decline when it cannot be represented."""
    flattened = numpy.asarray(result).reshape(-1)
    if dtype.kind == "integer":
        try:
            contiguous = numpy.asarray(flattened, dtype=numpy.dtype(dtype.name))
        except (OverflowError, TypeError, ValueError):
            return None
    else:
        try:
            flattened = numpy.asarray(flattened, dtype=numpy.float64)
        except (OverflowError, TypeError, ValueError):
            return None
        target_dtype = numpy.dtype(dtype.name)
        if target_dtype.itemsize < numpy.dtype(numpy.float64).itemsize:
            # A value the narrower dtype would round to infinity is not an
            # overflow the reference implementation produces; decline instead.
            finite = numpy.isfinite(flattened)
            outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
            if bool(numpy.any(finite & outside_range)):
                return None
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            contiguous = numpy.asarray(flattened, dtype=target_dtype)
    storage = NumPyStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage


def _finite_operands(*operands: Any) -> bool:
    """Return whether every operand is free of infinities and NaN."""
    finite = numpy.asarray(True)
    for operand in operands:
        finite = finite & numpy.all(numpy.isfinite(operand))
    return bool(finite)


def _shape_size(shape: tuple[int, ...]) -> int:
    return Shape.from_iterable(shape).size
