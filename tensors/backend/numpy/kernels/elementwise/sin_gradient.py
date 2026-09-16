"""NumPy implementation of the sine VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sin_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _view(grad).astype(numpy.float64, copy=False)
        values = _view(value).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    if bool(numpy.any(numpy.isinf(values))):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        derivative = numpy.cos(values)
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
