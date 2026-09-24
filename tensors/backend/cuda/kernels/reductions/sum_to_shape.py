"""CuPy implementation of summation down to a broadcast shape."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen
from tensors.backend.cuda.kernels.reductions.stability import _scaled_sum
from tensors.backend.cuda.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.cuda.kernels.reductions.stability import _sum_axes


def sum_to_shape(
    values: Any,
    input_shape: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    dtype: DataType,
) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    axes = _sum_axes(input_shape, shape)
    if axes is None:
        return None
    if dtype.kind == "integer":
        result = cupy.sum(values, axis=axes, keepdims=True, dtype=cupy.int64)
        return _storage(result.reshape(shape), dtype=dtype, output_shape=shape)
    values = _widen(values)
    direct = cupy.sum(values, axis=axes, keepdims=True)
    result = _scaled_sum(values, axes)
    finite_group = cupy.all(cupy.isfinite(values), axis=axes, keepdims=True)
    result = cupy.where(finite_group, result, direct)
    return _storage(
        cupy.asarray(result).reshape(shape), dtype=dtype, output_shape=shape
    )
