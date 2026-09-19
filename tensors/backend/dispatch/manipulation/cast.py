"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_cast(value: Tensor, *, dtype: DataType) -> Storage:
    """Run accelerated dtype conversion, or the Python reference."""
    from tensors.backend.python.kernels.manipulation.cast_tensor import (
        cast_tensor as reference,
    )

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(value, dtype=dtype)
    cast_tensor = _backend_kernel("cast_tensor")
    result = cast_tensor(value, dtype=dtype)
    if result is not None:
        return result
    return reference(value, dtype=dtype)
