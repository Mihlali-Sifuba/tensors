"""NumPy implementation of the variance VJP."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage


def reduce_variance_gradient(
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
    if values.size == 0:
        return _storage(values, dtype=dtype, output_shape=input_shape)
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
        scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized_values = values / safe_scale
        center = numpy.mean(normalized_values, axis=axes, keepdims=True)
        normalized = normalized_values - center
        result = expanded * normalized * scale * (2.0 / count)
    return _storage(result, dtype=dtype, output_shape=input_shape)
