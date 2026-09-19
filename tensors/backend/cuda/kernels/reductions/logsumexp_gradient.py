"""CuPy implementation of the stable log-sum-exp VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor
from tensors.backend.cuda.kernels.reductions.logsumexp_ops import _normalization_terms


def logsumexp_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    values = _working_values(value)
    upstream = _working_values(grad)
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
