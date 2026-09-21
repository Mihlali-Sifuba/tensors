"""NumPy implementation of log-softmax."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.kernels.reductions.logsumexp_ops import _normalization_terms

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def log_softmax(value: Tensor, axis: int, *, dtype: DataType) -> Storage | None:
    """Run fused softmax or log-softmax on finite values."""
    values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(values, axis)
    with _errstate(over="ignore", invalid="ignore"):
        result = values - maximum - correction
    if not _finite_operands(values, probabilities, result):
        return None
    return _storage(result, dtype=dtype, output_shape=value.shape)
