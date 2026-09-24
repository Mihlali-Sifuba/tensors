"""NumPy implementation of summation."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage
from tensors.backend.numpy.kernels.reductions.stability import _scaled_sum
from tensors.backend.numpy.kernels.reductions.stability import _summation_guard

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
    """Run a numerically guarded NumPy reduction."""
    if values.size == 0:
        return _storage(
            numpy.full(output_shape, 0), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    values = values.astype(numpy.float64, copy=False)
    if dtype.kind == "integer":
        result = numpy.sum(values, axis=axis, keepdims=keepdims, dtype=numpy.int64)
        return _storage(result, dtype=dtype, output_shape=output_shape)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        direct = numpy.sum(values, axis=axis, keepdims=True)
    ordinary = False
    minimum = numpy.min(values, axis=axis, keepdims=True)
    maximum = numpy.max(values, axis=axis, keepdims=True)
    ordinary = bool(
        numpy.all(
            numpy.isfinite(minimum)
            & numpy.isfinite(maximum)
            & numpy.isfinite(direct)
            & ((minimum >= 0.0) | (maximum <= 0.0))
        )
    )
    if ordinary:
        result = direct
    else:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = _scaled_sum(values, axis)
        safe = _summation_guard(values, axes=axis, keepdims=True)
        direct_safe = safe & numpy.all(numpy.isfinite(direct))
        result = numpy.where(direct_safe, direct, scaled)
        finite_group = numpy.all(numpy.isfinite(values), axis=axis, keepdims=True)
        result = numpy.where(finite_group, result, direct)
    if not keepdims and axis:
        result = numpy.squeeze(result, axis=axis)
    return _storage(result, dtype=dtype, output_shape=output_shape)
