"""NumPy implementation of the logistic function VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sigmoid_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    ``z / (1 + z) ** 2`` with ``z = exp(-|x|)``, which is the Python
    reference's form and is chosen for the same reason: evaluating the
    sigmoid and subtracting it from one gives exactly zero once the sigmoid
    rounds to one, around ``x = 20``, where the derivative is still about
    ``2e-9``. This form is symmetric in ``|x|`` and needs no branch.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        magnitude = numpy.exp(-numpy.abs(working))
        denominator = 1.0 + magnitude
        result = upstream * (magnitude / (denominator * denominator))
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sigmoid VJP kernel returned an unexpected result size")
    return storage
