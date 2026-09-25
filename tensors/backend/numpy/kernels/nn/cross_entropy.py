"""NumPy-native multiclass cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage
from tensors.backend.numpy.kernels.nn.losses import _reduce_losses
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms
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
    """Compute zero-safe dense cross-entropy from NumPy-native values."""
    if logits_shape != target_shape:
        return None
    values = numpy.asarray(logits_values).reshape(logits_shape).astype(numpy.float64)
    weights = numpy.asarray(target_values).reshape(target_shape).astype(numpy.float64)
    maximum, correction, _ = _normalization_terms(values, axis)
    all_negative_infinity = numpy.all(numpy.isneginf(values), axis=axis, keepdims=True)
    if bool(numpy.any(all_negative_infinity)):
        raise ValueError(
            "log_softmax is undefined when every value along an axis is -inf"
        )
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ordinary = values - maximum - correction
        positive = numpy.isposinf(values)
        positive_count = numpy.sum(positive, axis=axis, keepdims=True)
        infinity_logs = numpy.where(positive, -numpy.log(positive_count), -numpy.inf)
        log_probabilities = numpy.where(positive_count > 0, infinity_logs, ordinary)
        nan_group = numpy.any(numpy.isnan(values), axis=axis, keepdims=True)
        log_probabilities = numpy.where(nan_group, numpy.nan, log_probabilities)
        probability_dtype = numpy.dtype(
            logits_dtype.name if logits_dtype.kind == "floating" else "float64"
        )
        log_probabilities = log_probabilities.astype(probability_dtype).astype(
            numpy.float64
        )
        contributions = numpy.where(weights == 0.0, 0.0, -weights * log_probabilities)
        losses = numpy.sum(contributions, axis=axis)
    return _storage(
        _reduce_losses(losses, reduction),
        dtype=dtype,
        output_shape=output_shape,
    )
