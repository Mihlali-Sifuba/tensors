"""CuPy implementation of multiclass cross-entropy."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction
from tensors.backend.cuda.kernels.nn.losses import _reduce_losses


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
    values = _view(logits).astype(cupy.float64, copy=False)
    weights = _view(targets).astype(cupy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        log_probabilities = values - maximum - correction
        contributions = cupy.where(weights == 0.0, 0.0, -weights * log_probabilities)
        losses = cupy.sum(contributions, axis=axis)
    valid = (
        cupy.all(cupy.isfinite(values))
        & cupy.all(cupy.isfinite(weights))
        & cupy.all(cupy.isfinite(probabilities))
        & ~cupy.any(cupy.isnan(losses))
    )
    if not bool(valid):
        return None
    result = _reduce_losses(losses, reduction)
    return _storage(result, dtype=dtype, output_shape=output_shape)
