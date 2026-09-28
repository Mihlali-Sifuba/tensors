"""CUDA implementation of summation down to a broadcast shape."""

from __future__ import annotations

from typing import Any

import cupy

from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.kernels.reductions.exact import (
    certified_float_sum,
    exact_integer_sum,
)
from tensors.backend.cuda.kernels.reductions.stability import _sum_axes
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def sum_to_shape(
    values: Any,
    input_shape: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    dtype: DataType,
) -> Storage | None:
    """Reduce a broadcast gradient only when its native sum is conforming."""
    axes = _sum_axes(input_shape, shape)
    if axes is None:
        return None
    if values.size == 0:
        result = cupy.zeros(shape, dtype=cupy.dtype(dtype.name))
    elif dtype.kind == "integer":
        result = exact_integer_sum(values, axes, keepdims=True, dtype=dtype)
    else:
        result = certified_float_sum(values, axes)
    if result is None:
        return None
    return _storage(
        cupy.asarray(result).reshape(shape), dtype=dtype, output_shape=shape
    )
