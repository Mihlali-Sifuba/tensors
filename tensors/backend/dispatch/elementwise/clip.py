"""Dispatch for elementwise operations and their VJPs."""

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


def execute_clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage:
    """Run an elementwise clipping kernel."""
    from tensors.backend.python.kernels.elementwise.clip import clip as reference

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(value, min_value, max_value, dtype=dtype)
    clip = _backend_kernel("clip")
    result = clip(value, min_value, max_value, dtype=dtype)
    if result is not None:
        return result
    return reference(value, min_value, max_value, dtype=dtype)
