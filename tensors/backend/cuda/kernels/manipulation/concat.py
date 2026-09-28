"""CUDA-native concatenation along an existing axis."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def concat(
    values: Sequence[Any],
    shapes: Sequence[tuple[int, ...]],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Concatenate compact CUDA values into independently owned storage."""
    native_dtype = cupy.dtype(dtype.name)
    try:
        prepared = tuple(
            value.reshape(shape).astype(native_dtype, copy=False)
            for value, shape in zip(values, shapes)
        )
        result = (
            cupy.stack(prepared, axis=0)
            if not shapes[0]
            else cupy.concatenate(prepared, axis=axis)
        )
    except (TypeError, ValueError):
        return None
    if result.size != math.prod(output_shape):
        raise RuntimeError("concat kernel returned an unexpected result size")
    return CudaStorage(result, dtype)
