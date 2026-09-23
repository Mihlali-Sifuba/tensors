"""CUDA implementation of where."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def where(
    condition_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select between device data arrays using a device condition array."""
    if left_values.dtype == cupy.float32:
        left_values = _widen(left_values)
    if right_values.dtype == cupy.float32:
        right_values = _widen(right_values)
    result = cupy.where(condition_values != 0, left_values, right_values)
    narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("where kernel returned an unexpected result size")
    return storage
