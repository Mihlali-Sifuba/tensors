"""Dispatch for values built from parameters rather than transformed."""

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


def execute_linspace(
    start: int | float, stop: int | float, count: int, *, dtype: DataType
) -> Storage:
    """Create evenly spaced storage with an accelerated backend when safe."""
    from tensors.backend.python.kernels.creation.linspace import linspace as reference

    if not _array_work_is_large_enough(count, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(start, stop, count, dtype=dtype)
    linspace = _backend_kernel("linspace")
    result = linspace(start, stop, count, dtype=dtype)
    if result is not None:
        return result
    return reference(start, stop, count, dtype=dtype)
