"""Log-sum-exp and the shifted-exponential terms it shares.

``_normalization_terms`` lives here rather than under ``nn`` so that the
softmax family can depend on it one-way; the reverse would make the two
packages import each other."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _finite_operands, _numpy, _storage, _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def _normalization_terms(
    values: Any,
    axis: int | tuple[int, ...],
    numpy: Any,
) -> tuple[Any, Any, Any]:
    """Return stable maxima, corrections, and probabilities."""
    maximum = numpy.max(values, axis=axis, keepdims=True)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        probabilities = numpy.exp(deltas - correction)
    return maximum, correction, probabilities

def logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a stable log-sum-exp reduction on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axes,
        numpy,
    )
    with _errstate(numpy, over="ignore", invalid="ignore"):
        result = maximum + correction
    if not keepdims and axes:
        result = numpy.squeeze(result, axis=axes)
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def logsumexp_gradient(
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    _, _, probabilities = _normalization_terms(values, axes, numpy)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
    if not _finite_operands(
        values,
        upstream,
        probabilities,
        result,
        numpy=numpy,
    ):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )
