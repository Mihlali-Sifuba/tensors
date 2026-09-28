"""NumPy implementation of the arctan VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arctan_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arctan VJP on prepared native values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        reciprocal = 1.0 / numpy.abs(working)
        derivative = numpy.where(
            numpy.isinf(working),
            0.0,
            numpy.where(
                numpy.abs(working) <= 1.0,
                1.0 / (1.0 + working * working),
                reciprocal * reciprocal / (1.0 + reciprocal * reciprocal),
            ),
        )
        result = upstream * derivative
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("arctan VJP kernel returned an unexpected result size")
    return storage
