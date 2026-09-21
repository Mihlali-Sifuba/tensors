"""NumPy implementation of the SGD parameter update."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sgd_update(
    parameter: Tensor, gradient: Tensor, learning_rate: float
) -> Storage | None:
    """Apply one fused SGD update."""
    values = tensor_to_logical_array(parameter).astype(numpy.float64, copy=False)
    gradients = tensor_to_logical_array(gradient).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(result, dtype=parameter.dtype, output_shape=parameter.shape)
