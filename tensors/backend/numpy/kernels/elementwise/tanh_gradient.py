"""NumPy implementation of the tanh VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def tanh_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order tanh VJP on prepared native values."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = numpy.asarray(grad_values, dtype=numpy.float64)
        working = numpy.asarray(values, dtype=numpy.float64)
        magnitude = numpy.exp(-2.0 * numpy.abs(working))
        derivative = 4.0 * magnitude / (1.0 + magnitude) ** 2.0
        result = upstream * derivative
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("tanh VJP kernel returned an unexpected result size")
    return storage
