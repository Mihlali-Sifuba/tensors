"""NumPy implementation of the log-softmax VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def log_softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    upstream = _view(grad).astype(numpy.float64, copy=False)
    values = _view(value).astype(numpy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        total = numpy.sum(upstream, axis=axis, keepdims=True)
        result = upstream - probabilities * total
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
        & numpy.all(numpy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
