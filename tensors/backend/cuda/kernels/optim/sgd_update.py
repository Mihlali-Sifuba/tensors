"""CuPy implementation of the SGD parameter update."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sgd_update(
    parameter: Tensor, gradient: Tensor, learning_rate: float
) -> Storage | None:
    """Apply one fused SGD update."""
    values = _view(parameter).astype(cupy.float64, copy=False)
    gradients = _view(gradient).astype(cupy.float64, copy=False)
    if not _finite_operands(values, gradients):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(cupy.all(cupy.isfinite(result))):
        return None
    return _storage(result, dtype=parameter.dtype, output_shape=parameter.shape)
