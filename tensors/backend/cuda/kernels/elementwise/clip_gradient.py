"""CUDA implementation of the clip VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def clip_gradient(
    grad_values: Any,
    values: Any,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Pass gradients strictly inside the bounds and zero the boundaries."""
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(grad_values)
        working = _widen(values)
        mask = cupy.ones(output_shape, dtype=bool)
        if min_value is not None:
            mask &= working > min_value
        if max_value is not None:
            mask &= working < max_value
        result = cupy.where(
            cupy.isnan(working), cupy.nan, cupy.where(mask, upstream, 0.0)
        )
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("clip VJP kernel returned an unexpected result size")
    return storage
