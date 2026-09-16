"""CuPy implementation of the softmax VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    upstream = _view(grad).astype(cupy.float64, copy=False)
    values = _view(value).astype(cupy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        spread = cupy.max(upstream, axis=axis, keepdims=True) - cupy.min(
            upstream, axis=axis, keepdims=True
        )
        expectation = cupy.sum(upstream * probabilities, axis=axis, keepdims=True)
        result = probabilities * (upstream - expectation)
        result = cupy.where(spread == 0.0, 0.0, result)
    valid = (
        cupy.all(cupy.isfinite(values))
        & cupy.all(cupy.isfinite(upstream))
        & cupy.all(cupy.isfinite(probabilities))
        & ~cupy.any(cupy.max(probabilities, axis=axis) > 0.95)
        & cupy.all(cupy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
