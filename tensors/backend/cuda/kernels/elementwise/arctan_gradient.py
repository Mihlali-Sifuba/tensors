"""CUDA implementation of the arctan VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arctan_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arctan VJP on prepared device values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(grad_values)
        working = _widen(values)
        reciprocal = 1.0 / cupy.abs(working)
        derivative = cupy.where(
            cupy.isinf(working),
            0.0,
            cupy.where(
                cupy.abs(working) <= 1.0,
                1.0 / (1.0 + working * working),
                reciprocal * reciprocal / (1.0 + reciprocal * reciprocal),
            ),
        )
        result = upstream * derivative
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("arctan VJP kernel returned an unexpected result size")
    return storage
