"""CuPy implementation of the SGD parameter update."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sgd_update(
    parameter_values,
    gradient_values,
    learning_rate: float,
    *,
    dtype: DataType,
    shape: tuple[int, ...],
) -> Storage | None:
    """Apply one fused SGD update."""
    values = _widen(cupy.asarray(parameter_values))
    gradients = _widen(cupy.asarray(gradient_values))
    if not _finite_operands(values, gradients):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(cupy.all(cupy.isfinite(result))):
        return None
    return _storage(result, dtype=dtype, output_shape=shape)
