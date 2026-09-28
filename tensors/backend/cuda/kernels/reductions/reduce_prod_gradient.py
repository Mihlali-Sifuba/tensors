"""CuPy implementation of the product VJP."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen


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
    values = _widen(values)
    upstream = _widen(upstream)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(input_shape))
    )
    expanded = upstream.reshape(expanded_shape)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        is_nan = cupy.isnan(values)
        is_infinite = cupy.isinf(values)
        is_zero = values == 0.0
        ordinary = ~(is_nan | is_infinite | is_zero)
        nan_count = cupy.sum(is_nan, axis=axes, keepdims=True)
        infinity_count = cupy.sum(is_infinite, axis=axes, keepdims=True)
        zero_count = cupy.sum(is_zero, axis=axes, keepdims=True)
        other_nan = nan_count - is_nan
        other_infinity = infinity_count - is_infinite
        other_zero = zero_count - is_zero
        log_magnitude = cupy.sum(
            cupy.where(ordinary, cupy.log(cupy.abs(values)), 0.0),
            axis=axes,
            keepdims=True,
        )
        other_log_magnitude = log_magnitude - cupy.where(
            ordinary, cupy.log(cupy.abs(values)), 0.0
        )
        signs = cupy.copysign(1.0, values)
        sign = cupy.prod(signs, axis=axes, keepdims=True) * signs
        magnitude = cupy.exp(other_log_magnitude)
        magnitude = cupy.where(other_zero > 0, 0.0, magnitude)
        magnitude = cupy.where(other_infinity > 0, cupy.inf, magnitude)
        undefined = (other_nan > 0) | ((other_zero > 0) & (other_infinity > 0))
        guarded_derivative = cupy.where(undefined, cupy.nan, sign * magnitude)
        product = cupy.prod(values, axis=axes, keepdims=True)
        nonzero_product = cupy.prod(
            cupy.where(is_zero, 1.0, values), axis=axes, keepdims=True
        )
        direct_derivative = cupy.where(
            zero_count == 0,
            product / values,
            cupy.where((zero_count == 1) & is_zero, nonzero_product, 0.0),
        )
        unsafe = (~cupy.isfinite(direct_derivative)) | (
            (direct_derivative == 0.0) & (guarded_derivative != 0.0)
        )
        derivative = cupy.where(unsafe, guarded_derivative, direct_derivative)
        result = expanded * derivative
    return _storage(result, dtype=dtype, output_shape=input_shape)
