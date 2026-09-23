"""CUDA implementation of clip."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def clip(
    values: Any,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Clip device values while preserving equality, signed zero, and NaN."""
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = _widen(values) if values.dtype == cupy.float32 else values
        if min_value is not None:
            result = cupy.where(result < min_value, min_value, result)
        if max_value is not None:
            result = cupy.where(result > max_value, max_value, result)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("clip kernel returned an unexpected result size")
    return storage
