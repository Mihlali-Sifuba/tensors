"""CUDA-native log-softmax vector-Jacobian product."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _storage, _widen
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log_softmax_gradient(
    grad_values: Any,
    value_values: Any,
    grad_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    axis: int,
    *,
    dtype: DataType,
    value_dtype: DataType,
) -> Storage | None:
    """Apply the log-softmax Jacobian without dominant cancellation."""
    if grad_shape != value_shape:
        return None
    upstream = _widen(cupy.asarray(grad_values).reshape(grad_shape))
    values = _widen(cupy.asarray(value_values).reshape(value_shape))
    maximum, correction, raw_probabilities = _normalization_terms(values, axis)
    probability_dtype = cupy.dtype(
        value_dtype.name if value_dtype.kind == "floating" else "float64"
    )
    probabilities = _widen(_narrow(raw_probabilities, probability_dtype))
    if bool(cupy.any(cupy.all(cupy.isneginf(values), axis=axis, keepdims=True))):
        raise ValueError(
            "log_softmax gradient is undefined when every value along an axis is -inf"
        )
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        maxima = cupy.sum(values == maximum, axis=axis, keepdims=True)
        accurate_maximum_complement = -cupy.expm1(-correction)
        raw_complements = cupy.where(
            (maxima == 1) & cupy.isfinite(maximum) & (values == maximum),
            accurate_maximum_complement,
            1.0 - raw_probabilities,
        )
        finite_group = cupy.all(cupy.isfinite(values), axis=axis, keepdims=True)
        complements = cupy.where(finite_group, raw_complements, 1.0 - probabilities)
        moved_grad = cupy.moveaxis(upstream, axis, -1)
        moved_probabilities = cupy.moveaxis(probabilities, axis, -1)
        moved_complements = cupy.moveaxis(complements, axis, -1)
        scale = cupy.max(cupy.abs(moved_grad), axis=-1, keepdims=True)
        scalable = cupy.isfinite(scale) & (scale != 0.0)
        safe_scale = cupy.where(scalable, scale, 1.0)
        normalized = cupy.where(scalable, moved_grad / safe_scale, moved_grad)
        zero = cupy.zeros_like(normalized[..., :1])
        prefix = cupy.concatenate(
            (zero, cupy.cumsum(normalized, axis=-1)[..., :-1]), axis=-1
        )
        reversed_suffix = cupy.cumsum(normalized[..., ::-1], axis=-1)[..., ::-1]
        suffix = cupy.concatenate((reversed_suffix[..., 1:], zero), axis=-1)
        result = normalized * moved_complements - moved_probabilities * (
            prefix + suffix
        )
        result = cupy.where(scalable, result * safe_scale, result)
        result = cupy.moveaxis(result, -1, axis)
    return _storage(result, dtype=dtype, output_shape=value_shape)
