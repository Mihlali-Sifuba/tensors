"""NumPy implementation of the softplus VJP."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softplus_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    Softplus is the integral of the logistic function, so its derivative is
    that function exactly. The branch is the sigmoid kernel's, taken for the
    same range reason: each side is evaluated where its exponent is negative.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        magnitude = numpy.exp(-numpy.abs(working))
        derivative = numpy.where(
            working >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
        result = upstream * derivative
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Softplus VJP kernel returned an unexpected result size")
    return storage
