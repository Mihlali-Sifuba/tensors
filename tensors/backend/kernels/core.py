"""Shared provider boundary for the NumPy and CuPy kernels.

These helpers move values between Tensor/Storage and the native arrays the
kernel families operate on, and select the array module for the active
backend. Numerical helpers belong here only when several families need them."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from ...shape import Shape
from ..storage import CudaStorage, NumPyStorage, Storage, StorageKind

if TYPE_CHECKING:
    from ..._typing import Scalar
    from ...dtype import DataType
    from ...tensor import Tensor

def _view(tensor: Tensor, numpy: Any) -> Any:
    """Return compact logical values as a backend-native array.

    Provider kernels operate on compact arrays. Tensor metadata remains the
    source of truth, so a future non-compact layout is gathered before crossing
    this boundary rather than being reshaped as if logical and physical
    positions were identical.
    """
    storage = tensor._logical_storage_for(_array_kind(numpy))
    return storage.buffer.reshape(tensor.shape)

def _array_kind(numpy: Any) -> StorageKind:
    """Return the storage kind associated with an imported array provider."""
    return "cuda" if numpy.__name__.split(".", 1)[0] == "cupy" else "numpy"

def _cuda_integer(dtype: DataType) -> bool:
    """Return whether exact integer semantics require the Python fallback."""
    from ..config import get_backend

    return get_backend() == "cuda" and dtype.kind == "integer"

def _errstate(numpy: Any, **settings: str) -> Any:
    """Use NumPy warning controls when the selected array module provides them."""
    factory = getattr(numpy, "errstate", None)
    if factory is None:
        return nullcontext()
    return factory(**settings)

def _operand(value: Tensor | Scalar, dtype: DataType, numpy: Any) -> Any:
    """Return an array operand with Python-reference working precision."""
    from ...tensor import Tensor

    if _array_kind(numpy) == "cuda" and dtype.kind == "integer":
        # CuPy has no object dtype with Python's unbounded intermediate integer
        # semantics. Let the reference backend handle these operations.
        raise TypeError("CUDA integer kernels require the Python fallback")

    if isinstance(value, Tensor):
        result = _view(value, numpy)
    else:
        result = value
    working_dtype = numpy.float64 if dtype.kind == "floating" else object
    return numpy.asarray(result, dtype=working_dtype)

def _storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    numpy: Any,
) -> Storage | None:
    """Retain a native array result without changing tensor semantics."""
    backend = _array_kind(numpy)
    flattened = numpy.asarray(result).reshape(-1)
    if dtype.kind == "integer":
        if backend == "cuda":
            return None
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
            finite = numpy.isfinite(flattened)
            outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
            if bool(numpy.any(finite & outside_range)):
                return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            contiguous = numpy.asarray(flattened, dtype=target_dtype)
    storage: Storage
    if backend == "cuda":
        storage = CudaStorage(contiguous, dtype)
    else:
        storage = NumPyStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage

def _is_cuda() -> bool:
    from ..config import get_backend

    return get_backend() == "cuda"

def _finite_operands(*operands: Any, numpy: Any) -> bool:
    finite = numpy.asarray(True)
    for operand in operands:
        finite = finite & numpy.all(numpy.isfinite(operand))
    return bool(finite)

def _unsafe_finite_result(
    result: Any,
    *operands: Any,
    numpy: Any,
) -> bool:
    return _finite_operands(*operands, numpy=numpy) and not bool(
        numpy.all(numpy.isfinite(result))
    )

@lru_cache(maxsize=2)
def _import_array_module(module_name: str) -> Any:
    """Import and retain one array provider module."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        from ..config import BackendUnavailableError

        raise BackendUnavailableError(
            f"The {module_name} backend became unavailable after it was selected."
        ) from error

def _numpy() -> Any:
    """Return the cached array module selected by the active backend."""
    from ..config import get_backend

    backend = get_backend()
    return _import_array_module("cupy" if backend == "cuda" else "numpy")

def _shape_size(shape: tuple[int, ...]) -> int:
    return Shape.from_iterable(shape).size
