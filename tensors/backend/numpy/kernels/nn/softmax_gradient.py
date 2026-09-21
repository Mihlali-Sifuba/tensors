"""NumPy implementation of the softmax VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    upstream = tensor_to_logical_array(grad).astype(numpy.float64, copy=False)
    values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        spread = numpy.max(upstream, axis=axis, keepdims=True) - numpy.min(
            upstream, axis=axis, keepdims=True
        )
        expectation = numpy.sum(upstream * probabilities, axis=axis, keepdims=True)
        result = probabilities * (upstream - expectation)
        result = numpy.where(spread == 0.0, 0.0, result)
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
