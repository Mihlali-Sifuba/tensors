"""CUDA implementation of the sinh VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sinh_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order sinh VJP on prepared device values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(grad_values)
        working = _widen(values)
        derivative = cupy.cosh(working)
        result = upstream * derivative
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("sinh VJP kernel returned an unexpected result size")
    return storage
