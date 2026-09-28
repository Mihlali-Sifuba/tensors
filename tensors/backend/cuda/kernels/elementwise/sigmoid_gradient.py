"""CuPy implementation of the logistic function VJP."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sigmoid_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    ``z / (1 + z) ** 2`` with ``z = exp(-|x|)``, for the reason the other two
    backends use it: evaluating the sigmoid and subtracting it from one gives
    exactly zero once the sigmoid rounds to one, where the derivative is
    still representable.

    Both operands are widened and the result narrowed through PTX. Unlike the
    ReLU VJP, which only routes its upstream and can therefore leave it in
    binary32, this one multiplies by it, so a subnormal upstream would be
    flushed by the arithmetic rather than merely by the comparison.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(cupy.asarray(grad_values))
        working = _widen(cupy.asarray(values))
        magnitude = cupy.exp(-cupy.abs(working))
        denominator = 1.0 + magnitude
        result = upstream * (magnitude / (denominator * denominator))
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sigmoid VJP kernel returned an unexpected result size")
    return storage
