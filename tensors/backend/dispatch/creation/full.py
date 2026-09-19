"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
    _shape_size,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def execute_full(
    shape: tuple[int, ...], fill_value: int | float, *, dtype: DataType
) -> Storage:
    """Create constant-filled storage with an accelerated backend."""
    from tensors.backend.python.kernels.creation.full import full as reference

    if not _array_work_is_large_enough(_shape_size(shape), _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(shape, fill_value, dtype=dtype)
    full = _backend_kernel("full")
    result = full(shape, fill_value, dtype=dtype)
    if result is not None:
        return result
    return reference(shape, fill_value, dtype=dtype)
