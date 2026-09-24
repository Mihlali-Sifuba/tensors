"""NumPy implementation of the product VJP."""

from __future__ import annotations
import numpy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage as _storage


def reduce_prod_gradient(
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
    expanded = upstream.reshape(expanded_shape)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        is_nan = numpy.isnan(values)
        is_infinite = numpy.isinf(values)
        is_zero = values == 0.0
        ordinary = ~(is_nan | is_infinite | is_zero)
        nan_count = numpy.sum(is_nan, axis=axes, keepdims=True)
        infinity_count = numpy.sum(is_infinite, axis=axes, keepdims=True)
        zero_count = numpy.sum(is_zero, axis=axes, keepdims=True)
        other_nan = nan_count - is_nan
        other_infinity = infinity_count - is_infinite
        other_zero = zero_count - is_zero
        log_magnitude = numpy.sum(
            numpy.where(ordinary, numpy.log(numpy.abs(values)), 0.0),
            axis=axes,
            keepdims=True,
        )
        other_log_magnitude = log_magnitude - numpy.where(
            ordinary, numpy.log(numpy.abs(values)), 0.0
        )
        signs = numpy.copysign(1.0, values)
        sign = numpy.prod(signs, axis=axes, keepdims=True) * signs
        magnitude = numpy.exp(other_log_magnitude)
        magnitude = numpy.where(other_zero > 0, 0.0, magnitude)
        magnitude = numpy.where(other_infinity > 0, numpy.inf, magnitude)
        undefined = (other_nan > 0) | ((other_zero > 0) & (other_infinity > 0))
        guarded_derivative = numpy.where(undefined, numpy.nan, sign * magnitude)
        product = numpy.prod(values, axis=axes, keepdims=True)
        nonzero_product = numpy.prod(
            numpy.where(is_zero, 1.0, values), axis=axes, keepdims=True
        )
        direct_derivative = numpy.where(
            zero_count == 0,
            product / values,
            numpy.where((zero_count == 1) & is_zero, nonzero_product, 0.0),
        )
        unsafe = (~numpy.isfinite(direct_derivative)) | (
            (direct_derivative == 0.0) & (guarded_derivative != 0.0)
        )
        derivative = numpy.where(unsafe, guarded_derivative, direct_derivative)
        result = expanded * derivative
    return _storage(result, dtype=dtype, output_shape=input_shape)
