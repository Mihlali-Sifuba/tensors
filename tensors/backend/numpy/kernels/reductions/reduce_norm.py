"""NumPy implementation of the Euclidean norm."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_norm(
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
            numpy.full(output_shape, 0.0), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = values.astype(numpy.float64, copy=False)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        absolute = numpy.abs(values)
        scale = numpy.max(absolute, axis=axis, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        normalized_magnitude = numpy.sqrt(
            numpy.sum(normalized * normalized, axis=axis, keepdims=keepdims)
        )
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        result = output_scale * normalized_magnitude
    has_nan = numpy.any(numpy.isnan(values), axis=axis, keepdims=keepdims)
    has_infinity = numpy.any(numpy.isinf(values), axis=axis, keepdims=keepdims)
    result = numpy.where(
        has_nan, numpy.nan, numpy.where(has_infinity, numpy.inf, result)
    )
    return _storage(result, dtype=dtype, output_shape=output_shape)
