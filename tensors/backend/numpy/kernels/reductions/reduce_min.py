"""NumPy implementation of the minimum."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_min(
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if values.size == 0:
        raise ValueError("Cannot compute min of empty tensor")
    remaining = tuple(index for index in range(len(input_shape)) if index not in axes)
    permutation = remaining + axes
    outer_shape = tuple(input_shape[index] for index in remaining)
    reduction_size = 1
    for index in axes:
        reduction_size *= input_shape[index]
    grouped = numpy.transpose(values, permutation).reshape(
        outer_shape + (reduction_size,)
    )
    extrema = numpy.min(grouped, axis=-1, keepdims=True)
    nan_group = numpy.any(numpy.isnan(grouped), axis=-1, keepdims=True)
    candidates = numpy.where(nan_group, numpy.isnan(grouped), grouped == extrema)
    first = numpy.argmax(candidates, axis=-1, keepdims=True)
    result = numpy.take_along_axis(grouped, first, axis=-1).reshape(outer_shape)
    if keepdims:
        kept_shape = tuple(
            1 if index in axes else input_shape[index]
            for index in range(len(input_shape))
        )
        result = result.reshape(kept_shape)
    return _storage(result, dtype=dtype, output_shape=output_shape)
