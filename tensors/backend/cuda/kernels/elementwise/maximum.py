"""CUDA implementation of maximum."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def maximum(
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select larger values, propagating NaN and choosing the left tie."""
    if left_values.dtype == cupy.float32:
        left_values = _widen(left_values)
    if right_values.dtype == cupy.float32:
        right_values = _widen(right_values)
    result = cupy.where(
        cupy.isnan(left_values),
        left_values,
        cupy.where(
            cupy.isnan(right_values),
            right_values,
            cupy.where(left_values >= right_values, left_values, right_values),
        ),
    )
    narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("maximum kernel returned an unexpected result size")
    return storage
