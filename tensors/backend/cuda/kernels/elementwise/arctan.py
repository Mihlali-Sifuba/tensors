"""CUDA implementation of arctan."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arctan(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate arctan on prepared device values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if values.dtype.kind in "iu":
            values = values.astype(cupy.float64, copy=False)
        working = _widen(values)
        result = cupy.arctan(working)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("arctan kernel returned an unexpected result size")
    return storage
