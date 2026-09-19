"""CuPy implementation of the square root VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sqrt_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _working_values(grad)
        values = _working_values(value)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    if bool(cupy.any(values == 0.0)):
        raise ValueError("sqrt derivative is undefined at zero")
    if bool(cupy.any(values < 0.0)):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        derivative = 1.0 / (2.0 * cupy.sqrt(values))
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
