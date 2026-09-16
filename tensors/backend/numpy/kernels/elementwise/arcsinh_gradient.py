"""NumPy implementation of the inverse hyperbolic sine VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def arcsinh_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _view(grad).astype(numpy.float64, copy=False)
        values = _view(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        reciprocal = 1.0 / numpy.abs(values)
        derivative = numpy.where(
            numpy.isinf(values),
            0.0,
            numpy.where(
                numpy.abs(values) <= 1.0,
                1.0 / numpy.sqrt(1.0 + values * values),
                reciprocal / numpy.sqrt(1.0 + reciprocal * reciprocal),
            ),
        )
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
