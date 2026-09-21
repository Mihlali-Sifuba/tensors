"""NumPy implementation of absolute value."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def abs(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    ``numpy.abs`` computes in the dtype it is given, so an integer array stays
    integral and a large ``int64`` is never routed through ``float64``. The
    one integer input it would answer wrongly — a signed dtype's least value,
    which it wraps back to itself — has already been refused by
    `execute_abs`. Host arithmetic has no flush-to-zero, so a subnormal
    operand keeps its value.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.abs(values)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Abs kernel returned an unexpected result size")
    return storage
