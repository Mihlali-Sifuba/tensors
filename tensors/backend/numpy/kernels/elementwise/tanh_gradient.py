"""NumPy implementation of the hyperbolic tangent VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def tanh_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = tensor_to_logical_array(grad).astype(numpy.float64, copy=False)
        values = tensor_to_logical_array(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        magnitude = numpy.exp(-2.0 * numpy.abs(values))
        derivative = 4.0 * magnitude / (1.0 + magnitude) ** 2.0
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
