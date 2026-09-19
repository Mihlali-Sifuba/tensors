"""NumPy implementation of multiclass cross-entropy."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction
from tensors.backend.numpy.kernels.nn.losses import _reduce_losses


def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused dense multiclass cross-entropy."""
    values = _view(logits).astype(numpy.float64, copy=False)
    weights = _view(targets).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        log_probabilities = values - maximum - correction
        contributions = numpy.where(weights == 0.0, 0.0, -weights * log_probabilities)
        losses = numpy.sum(contributions, axis=axis)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.isnan(losses))
    )
    if not bool(valid):
        return None
    result = _reduce_losses(losses, reduction)
    return _storage(result, dtype=dtype, output_shape=output_shape)
