"""CuPy implementation of the multiclass cross-entropy VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _shape_size
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms

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
    values = _view(logits).astype(cupy.float64, copy=False)
    weights = _view(targets).astype(cupy.float64, copy=False)
    upstream = _view(grad).astype(cupy.float64, copy=False)
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
                cupy.broadcast_to(upstream.reshape(()), sample_shape) * scale
            )
        expanded_upstream = cupy.expand_dims(expanded_upstream, axis=axis)
    except (TypeError, ValueError):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        target_mass = cupy.sum(weights, axis=axis, keepdims=True)
        log_probabilities = values - maximum - correction
        logits_result = expanded_upstream * (target_mass * probabilities - weights)
        targets_result = -expanded_upstream * log_probabilities
        zero_upstream = expanded_upstream == 0.0
        logits_result = cupy.where(zero_upstream, 0.0, logits_result)
        targets_result = cupy.where(zero_upstream, 0.0, targets_result)
    valid = (
        cupy.all(cupy.isfinite(values))
        & cupy.all(cupy.isfinite(weights))
        & cupy.all(cupy.isfinite(upstream))
        & cupy.all(cupy.isfinite(probabilities))
        & ~cupy.any(cupy.max(probabilities, axis=axis) > 0.95)
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
