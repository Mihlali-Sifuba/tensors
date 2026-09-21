"""NumPy implementation of the stable log-sum-exp."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms


def logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a stable log-sum-exp reduction on finite values."""
    values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(values, axes)
    with _errstate(over="ignore", invalid="ignore"):
        result = maximum + correction
    if not keepdims and axes:
        result = numpy.squeeze(result, axis=axes)
    if not _finite_operands(values, probabilities, result):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
