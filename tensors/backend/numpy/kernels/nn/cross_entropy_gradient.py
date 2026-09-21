"""NumPy implementation of the multiclass cross-entropy VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _shape_size
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction


def cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested dense multiclass cross-entropy VJPs."""
    need_logits, need_targets = needs_input_grad
    values = tensor_to_logical_array(logits).astype(numpy.float64, copy=False)
    weights = tensor_to_logical_array(targets).astype(numpy.float64, copy=False)
    upstream = tensor_to_logical_array(grad).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(values, axis)
    sample_shape = values.shape[:axis] + values.shape[axis + 1 :]
    try:
        if reduction == "none":
            expanded_upstream = upstream.reshape(sample_shape)
        else:
            scale = (
                1.0 / _shape_size(sample_shape)
                if reduction == "mean" and _shape_size(sample_shape)
                else 1.0
            )
            expanded_upstream = (
                numpy.broadcast_to(upstream.reshape(()), sample_shape) * scale
            )
        expanded_upstream = numpy.expand_dims(expanded_upstream, axis=axis)
    except (TypeError, ValueError):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        target_mass = numpy.sum(weights, axis=axis, keepdims=True)
        log_probabilities = values - maximum - correction
        logits_result = expanded_upstream * (target_mass * probabilities - weights)
        targets_result = -expanded_upstream * log_probabilities
        zero_upstream = expanded_upstream == 0.0
        logits_result = numpy.where(zero_upstream, 0.0, logits_result)
        targets_result = numpy.where(zero_upstream, 0.0, targets_result)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
    )
    if not bool(valid):
        return None
    logits_storage = None
    if need_logits:
        logits_storage = _storage(
            logits_result, dtype=grad.dtype, output_shape=logits.shape
        )
        if logits_storage is None:
            return None
    targets_storage = None
    if need_targets:
        targets_storage = _storage(
            targets_result, dtype=grad.dtype, output_shape=targets.shape
        )
        if targets_storage is None:
            return None
    return (logits_storage, targets_storage)
