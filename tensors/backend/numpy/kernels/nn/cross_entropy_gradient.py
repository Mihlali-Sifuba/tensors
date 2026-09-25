"""NumPy-native multiclass cross-entropy VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _shape_size, _storage
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms
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
    """Compute cancellation-resistant VJPs from NumPy-native values."""
    if logits_shape != target_shape:
        return None
    values = numpy.asarray(logits_values).reshape(logits_shape).astype(numpy.float64)
    weights = numpy.asarray(target_values).reshape(target_shape).astype(numpy.float64)
    upstream = numpy.asarray(grad_values).reshape(grad_shape).astype(numpy.float64)
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
            numpy.broadcast_to(upstream.reshape(()), sample_shape) * scale
        )
    expanded_upstream = numpy.expand_dims(expanded_upstream, axis=axis)
    maximum, correction, raw_probabilities = _normalization_terms(values, axis)
    if bool(numpy.any(numpy.all(numpy.isneginf(values), axis=axis, keepdims=True))):
        raise ValueError(
            "cross-entropy gradient is undefined when every value along an axis is -inf"
        )
    probability_dtype = numpy.dtype(
        logits_dtype.name if logits_dtype.kind == "floating" else "float64"
    )
    probabilities = raw_probabilities.astype(probability_dtype).astype(numpy.float64)
    need_logits, need_targets = needs_input_grad
    logits_result = None
    targets_result = None
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        zero_upstream = expanded_upstream == 0.0
        if need_logits:
            maxima = numpy.sum(values == maximum, axis=axis, keepdims=True)
            accurate_maximum_complement = -numpy.expm1(-correction)
            raw_complements = numpy.where(
                (maxima == 1) & numpy.isfinite(maximum) & (values == maximum),
                accurate_maximum_complement,
                1.0 - raw_probabilities,
            )
            finite_group = numpy.all(numpy.isfinite(values), axis=axis, keepdims=True)
            complements = numpy.where(
                finite_group, raw_complements, 1.0 - probabilities
            )
            target_mass = numpy.sum(weights, axis=axis, keepdims=True)
            ordinary = target_mass * probabilities - weights
            dominant = (target_mass - weights) - target_mass * complements
            derivative = numpy.where(probabilities > 0.5, dominant, ordinary)
            logits_result = numpy.where(
                zero_upstream, 0.0, expanded_upstream * derivative
            )
        if need_targets:
            ordinary_logs = values - maximum - correction
            positive = numpy.isposinf(values)
            positive_count = numpy.sum(positive, axis=axis, keepdims=True)
            infinity_logs = numpy.where(
                positive, -numpy.log(positive_count), -numpy.inf
            )
            log_probabilities = numpy.where(
                positive_count > 0, infinity_logs, ordinary_logs
            )
            nan_group = numpy.any(numpy.isnan(values), axis=axis, keepdims=True)
            log_probabilities = numpy.where(nan_group, numpy.nan, log_probabilities)
            targets_result = numpy.where(
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
