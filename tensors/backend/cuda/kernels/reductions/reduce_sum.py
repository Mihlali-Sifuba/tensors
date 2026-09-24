"""CUDA implementation of summation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.kernels.reductions.exact import (
    certified_float_sum,
    exact_integer_sum,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_sum(
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return a conforming native sum, declining uncertified float groups."""
    if values.size == 0:
        return _storage(
            cupy.full(output_shape, 0, dtype=cupy.dtype(dtype.name)),
            dtype=dtype,
            output_shape=output_shape,
        )
    if dtype.kind == "integer":
        result = exact_integer_sum(values, axes, keepdims=keepdims, dtype=dtype)
    else:
        result = certified_float_sum(values, axes)
        if result is not None and not keepdims and axes:
            result = cupy.squeeze(result, axis=axes)
    if result is None:
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
