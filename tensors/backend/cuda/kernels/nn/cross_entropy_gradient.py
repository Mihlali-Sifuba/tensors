"""CUDA-native multiclass cross-entropy VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import (
    _errstate,
    _narrow,
    _shape_size,
    _storage,
    _widen,
)
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def cross_entropy_gradient(
    grad_values: Any,
    logits_values: Any,
    target_values: Any,
    grad_shape: tuple[int, ...],
    logits_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    logits_dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Compute cancellation-resistant VJPs from device-native values."""
    if logits_shape != target_shape:
        return None
    values = _widen(cupy.asarray(logits_values).reshape(logits_shape))
    weights = _widen(cupy.asarray(target_values).reshape(target_shape))
    upstream = _widen(cupy.asarray(grad_values).reshape(grad_shape))
    sample_shape = logits_shape[:axis] + logits_shape[axis + 1 :]
    if reduction == "none":
        if grad_shape != sample_shape and not (
            sample_shape == () and grad_shape == (1,)
        ):
            return None
        expanded_upstream = upstream.reshape(sample_shape)
    else:
        if upstream.size != 1:
            return None
        sample_size = _shape_size(sample_shape)
        scale = 1.0 / sample_size if reduction == "mean" and sample_size else 1.0
        expanded_upstream = (
            cupy.broadcast_to(upstream.reshape(()), sample_shape) * scale
        )
    expanded_upstream = cupy.expand_dims(expanded_upstream, axis=axis)
    maximum, correction, raw_probabilities = _normalization_terms(values, axis)
    if bool(cupy.any(cupy.all(cupy.isneginf(values), axis=axis, keepdims=True))):
        raise ValueError(
            "cross-entropy gradient is undefined when every value along an axis is -inf"
        )
    probability_dtype = cupy.dtype(
        logits_dtype.name if logits_dtype.kind == "floating" else "float64"
    )
    probabilities = _widen(_narrow(raw_probabilities, probability_dtype))
    need_logits, need_targets = needs_input_grad
    logits_result = None
    targets_result = None
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        zero_upstream = expanded_upstream == 0.0
        if need_logits:
            maxima = cupy.sum(values == maximum, axis=axis, keepdims=True)
            accurate_maximum_complement = -cupy.expm1(-correction)
            raw_complements = cupy.where(
                (maxima == 1) & cupy.isfinite(maximum) & (values == maximum),
                accurate_maximum_complement,
                1.0 - raw_probabilities,
            )
            finite_group = cupy.all(cupy.isfinite(values), axis=axis, keepdims=True)
            complements = cupy.where(finite_group, raw_complements, 1.0 - probabilities)
            target_mass = cupy.sum(weights, axis=axis, keepdims=True)
            ordinary = target_mass * probabilities - weights
            dominant = (target_mass - weights) - target_mass * complements
            derivative = cupy.where(probabilities > 0.5, dominant, ordinary)
            logits_result = cupy.where(
                zero_upstream, 0.0, expanded_upstream * derivative
            )
        if need_targets:
            ordinary_logs = values - maximum - correction
            positive = cupy.isposinf(values)
            positive_count = cupy.sum(positive, axis=axis, keepdims=True)
            infinity_logs = cupy.where(positive, -cupy.log(positive_count), -cupy.inf)
            log_probabilities = cupy.where(
                positive_count > 0, infinity_logs, ordinary_logs
            )
            nan_group = cupy.any(cupy.isnan(values), axis=axis, keepdims=True)
            log_probabilities = cupy.where(nan_group, cupy.nan, log_probabilities)
            targets_result = cupy.where(
                zero_upstream, 0.0, -expanded_upstream * log_probabilities
            )
    logits_storage = (
        _storage(logits_result, dtype=dtype, output_shape=logits_shape)
        if need_logits
        else None
    )
    targets_storage = (
        _storage(targets_result, dtype=dtype, output_shape=target_shape)
        if need_targets
        else None
    )
    if (need_logits and logits_storage is None) or (
        need_targets and targets_storage is None
    ):
        return None
    return logits_storage, targets_storage
