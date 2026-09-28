"""NumPy implementation of the SGD parameter update."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sgd_update(
    parameter_values,
    gradient_values,
    learning_rate: float,
    *,
    dtype: DataType,
    shape: tuple[int, ...],
) -> Storage | None:
    """Apply one fused SGD update."""
    values = numpy.asarray(parameter_values).astype(numpy.float64, copy=False)
    gradients = numpy.asarray(gradient_values).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(result, dtype=dtype, output_shape=shape)
