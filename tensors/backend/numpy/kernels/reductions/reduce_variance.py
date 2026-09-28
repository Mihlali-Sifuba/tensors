"""NumPy implementation of the variance."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_variance(
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
        return _storage(
            numpy.full(output_shape, numpy.nan), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = values.astype(numpy.float64, copy=False)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = numpy.max(numpy.abs(values), axis=axis, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized_values = values / safe_scale
        center = numpy.mean(normalized_values, axis=axis, keepdims=True)
        normalized = normalized_values - center
        normalized_variance = numpy.mean(
            normalized * normalized, axis=axis, keepdims=keepdims
        )
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        deviation = output_scale * numpy.sqrt(normalized_variance)
        result = deviation * deviation
    return _storage(result, dtype=dtype, output_shape=output_shape)
