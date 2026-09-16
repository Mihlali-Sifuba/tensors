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


def execute_eye(rows: int, columns: int, k: int, *, dtype: DataType) -> Storage:
    """Create identity-like matrix storage with an accelerated backend."""
    from tensors.backend.python.kernels.creation.eye import eye as reference

    shape = (rows, columns)
    if not _array_work_is_large_enough(_shape_size(shape), _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(rows, columns, k, dtype=dtype)
    eye = _backend_kernel("eye")
    result = eye(rows, columns, k, dtype=dtype)
    if result is not None:
        return result
    return reference(rows, columns, k, dtype=dtype)
