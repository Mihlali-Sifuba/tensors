"""CuPy implementation of the softplus VJP."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softplus_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    Softplus is the integral of the logistic function, so its derivative is
    that function exactly, taking each side where its exponent is negative.

    Both operands are widened and the result narrowed through PTX, because
    this VJP multiplies by its upstream gradient rather than routing it.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(cupy.asarray(grad_values))
        working = _widen(cupy.asarray(values))
        magnitude = cupy.exp(-cupy.abs(working))
        derivative = cupy.where(
            working >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
        )
        result = upstream * derivative
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Softplus VJP kernel returned an unexpected result size")
    return storage
