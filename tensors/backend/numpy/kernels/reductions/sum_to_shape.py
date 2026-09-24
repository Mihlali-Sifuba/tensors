"""NumPy implementation of summation down to a broadcast shape."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage
from tensors.backend.numpy.kernels.reductions.stability import _scaled_sum
from tensors.backend.numpy.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.numpy.kernels.reductions.stability import _sum_axes


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
        result = numpy.sum(values, axis=axes, keepdims=True, dtype=numpy.int64)
        return _storage(result.reshape(shape), dtype=dtype, output_shape=shape)
    values = values.astype(numpy.float64, copy=False)
    direct = numpy.sum(values, axis=axes, keepdims=True)
    result = _scaled_sum(values, axes)
    finite_group = numpy.all(numpy.isfinite(values), axis=axes, keepdims=True)
    result = numpy.where(finite_group, result, direct)
    return _storage(
        numpy.asarray(result).reshape(shape), dtype=dtype, output_shape=shape
    )
