"""CuPy implementation of the inverse hyperbolic sine VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def arcsinh_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _view(grad).astype(cupy.float64, copy=False)
        values = _view(value).astype(cupy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        reciprocal = 1.0 / cupy.abs(values)
        derivative = cupy.where(
            cupy.isinf(values),
            0.0,
            cupy.where(
                cupy.abs(values) <= 1.0,
                1.0 / cupy.sqrt(1.0 + values * values),
                reciprocal / cupy.sqrt(1.0 + reciprocal * reciprocal),
            ),
        )
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
