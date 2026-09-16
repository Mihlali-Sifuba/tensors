"""CuPy implementation of the tangent VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def tan_gradient(grad: Tensor, value: Tensor) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    try:
        upstream = _view(grad).astype(cupy.float64, copy=False)
        values = _view(value).astype(cupy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None
    if bool(cupy.any(cupy.isinf(values))):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        cosine = cupy.cos(values)
        if bool(cupy.any(cupy.abs(cosine) < cupy.finfo(cupy.float64).eps)):
            return None
        derivative = 1.0 / (cosine * cosine)
        result = upstream * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
