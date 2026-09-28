"""NumPy-native softmax vector-Jacobian product."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softmax_gradient(
    grad_values: Any,
    value_values: Any,
    grad_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
    value_dtype: DataType,
) -> Storage | None:
    """Apply the softmax Jacobian without dominant cancellation."""
    if grad_shape != value_shape:
        return None
    upstream = numpy.asarray(grad_values).reshape(grad_shape).astype(numpy.float64)
    values = numpy.asarray(value_values).reshape(value_shape).astype(numpy.float64)
    maximum, correction, raw_probabilities = _normalization_terms(values, axis)
    probability_dtype = numpy.dtype(
        value_dtype.name if value_dtype.kind == "floating" else "float64"
    )
    probabilities = raw_probabilities.astype(probability_dtype).astype(numpy.float64)
    if bool(numpy.any(numpy.all(numpy.isneginf(values), axis=axis, keepdims=True))):
        raise ValueError(
            "softmax gradient is undefined when every value along an axis is -inf"
        )
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        maxima = numpy.sum(values == maximum, axis=axis, keepdims=True)
        accurate_maximum_complement = -numpy.expm1(-correction)
        raw_complements = numpy.where(
            (maxima == 1) & numpy.isfinite(maximum) & (values == maximum),
            accurate_maximum_complement,
            1.0 - raw_probabilities,
        )
        finite_group = numpy.all(numpy.isfinite(values), axis=axis, keepdims=True)
        complements = numpy.where(finite_group, raw_complements, 1.0 - probabilities)
        moved_grad = numpy.moveaxis(upstream, axis, -1)
        moved_probabilities = numpy.moveaxis(probabilities, axis, -1)
        moved_complements = numpy.moveaxis(complements, axis, -1)
        scale = numpy.max(numpy.abs(moved_grad), axis=-1, keepdims=True)
        scalable = numpy.isfinite(scale) & (scale != 0.0)
        safe_scale = numpy.where(scalable, scale, 1.0)
        normalized = numpy.where(scalable, moved_grad / safe_scale, moved_grad)
        weighted = normalized * moved_probabilities
        zero = numpy.zeros_like(weighted[..., :1])
        prefix = numpy.concatenate(
            (zero, numpy.cumsum(weighted, axis=-1)[..., :-1]), axis=-1
        )
        reversed_suffix = numpy.cumsum(weighted[..., ::-1], axis=-1)[..., ::-1]
        suffix = numpy.concatenate((reversed_suffix[..., 1:], zero), axis=-1)
        centered = normalized * moved_complements - prefix - suffix
        centered = numpy.where(scalable, centered * safe_scale, centered)
        rounded = centered.astype(numpy.dtype(dtype.name)).astype(numpy.float64)
        result = moved_probabilities * rounded
        spread = numpy.max(moved_grad, axis=-1, keepdims=True) - numpy.min(
            moved_grad, axis=-1, keepdims=True
        )
        result = numpy.where(spread == 0.0, 0.0, result)
        result = numpy.moveaxis(result, -1, axis)
    return _storage(result, dtype=dtype, output_shape=value_shape)
