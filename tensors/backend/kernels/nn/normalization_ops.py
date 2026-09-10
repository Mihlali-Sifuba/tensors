"""Softmax and log-softmax, and their vector-Jacobian products."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _finite_operands, _numpy, _storage, _view
from ..reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import NormalizationOperation

def normalization(
    operation: NormalizationOperation,
    value: Tensor,
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run fused softmax or log-softmax on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    if operation == "softmax":
        result = probabilities
    else:
        with _errstate(numpy, over="ignore", invalid="ignore"):
            result = values - maximum - correction
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def normalization_gradient(
    operation: NormalizationOperation,
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis, numpy)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        if operation == "softmax":
            spread = numpy.max(upstream, axis=axis, keepdims=True) - numpy.min(
                upstream,
                axis=axis,
                keepdims=True,
            )
            expectation = numpy.sum(
                upstream * probabilities,
                axis=axis,
                keepdims=True,
            )
            result = probabilities * (upstream - expectation)
            result = numpy.where(spread == 0.0, 0.0, result)
        else:
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
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )
