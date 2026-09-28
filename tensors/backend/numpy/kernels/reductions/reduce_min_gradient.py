"""NumPy implementation of the minimum VJP."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage


def reduce_min_gradient(
    upstream: Any,
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = values.astype(numpy.float64, copy=False)
    upstream = upstream.astype(numpy.float64, copy=False)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(input_shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= input_shape[axis]
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        has_nan = numpy.any(numpy.isnan(values), axis=axes, keepdims=True)
        function = numpy.min
        extreme = function(values, axis=axes, keepdims=True)
        selected = values == extreme
        ties = numpy.sum(selected, axis=axes, keepdims=True)
        result = numpy.where(has_nan, numpy.nan, expanded * selected / ties)
    return _storage(result, dtype=dtype, output_shape=input_shape)
