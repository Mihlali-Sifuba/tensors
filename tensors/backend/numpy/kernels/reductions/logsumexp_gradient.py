"""NumPy implementation of the stable log-sum-exp VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms


def logsumexp_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    values = _view(value).astype(numpy.float64, copy=False)
    upstream = _view(grad).astype(numpy.float64, copy=False)
    _, _, probabilities = _normalization_terms(values, axes)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(value.shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
    if not _finite_operands(values, upstream, probabilities, result):
        return None
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
