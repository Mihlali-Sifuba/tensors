"""CuPy implementation of the inverse hyperbolic tangent VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def arctanh_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _working_values(grad)
        values = _working_values(value)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        derivative = 1.0 / (1.0 - values * values)
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
