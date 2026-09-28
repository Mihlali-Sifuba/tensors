"""CUDA-native multiclass cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _narrow, _storage, _widen
from tensors.backend.cuda.kernels.nn.losses import _reduce_losses
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def cross_entropy(
    logits_values: Any,
    target_values: Any,
    logits_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    logits_dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Compute zero-safe dense cross-entropy from device-native values."""
    if logits_shape != target_shape:
        return None
    values = _widen(cupy.asarray(logits_values).reshape(logits_shape))
    weights = _widen(cupy.asarray(target_values).reshape(target_shape))
    maximum, correction, _ = _normalization_terms(values, axis)
    all_negative_infinity = cupy.all(cupy.isneginf(values), axis=axis, keepdims=True)
    if bool(cupy.any(all_negative_infinity)):
        raise ValueError(
            "log_softmax is undefined when every value along an axis is -inf"
        )
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ordinary = values - maximum - correction
        positive = cupy.isposinf(values)
        positive_count = cupy.sum(positive, axis=axis, keepdims=True)
        infinity_logs = cupy.where(positive, -cupy.log(positive_count), -cupy.inf)
        log_probabilities = cupy.where(positive_count > 0, infinity_logs, ordinary)
        nan_group = cupy.any(cupy.isnan(values), axis=axis, keepdims=True)
        log_probabilities = cupy.where(nan_group, cupy.nan, log_probabilities)
        probability_dtype = cupy.dtype(
            logits_dtype.name if logits_dtype.kind == "floating" else "float64"
        )
        log_probabilities = _widen(_narrow(log_probabilities, probability_dtype))
        contributions = cupy.where(weights == 0.0, 0.0, -weights * log_probabilities)
        losses = cupy.sum(contributions, axis=axis)
    return _storage(
        _reduce_losses(losses, reduction),
        dtype=dtype,
        output_shape=output_shape,
    )
