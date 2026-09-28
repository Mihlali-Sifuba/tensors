"""CuPy implementation of the exponential VJP."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def exp_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    Exp is its own derivative, so this evaluates the forward's function and
    multiplies, in binary64, narrowing once. Both operands are widened
    through PTX: this VJP multiplies by its upstream gradient rather than
    routing it, so a subnormal upstream would otherwise be flushed by the
    arithmetic and not merely by a comparison.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(grad_values)
        working = _widen(values)
        result = upstream * cupy.exp(working)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Exp VJP kernel returned an unexpected result size")
    return storage
