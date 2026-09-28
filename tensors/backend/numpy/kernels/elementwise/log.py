"""NumPy implementation of the natural logarithm."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    The domain is settled before this runs, in the dispatcher, so nothing
    here inspects a value: every operand reaching this point is positive or
    NaN, and ``numpy.log`` returns NaN for NaN and ``+inf`` for ``+inf``.

    Evaluated in binary64 and rounded once on the way into storage, which is
    what the Python reference does. Host arithmetic has no flush-to-zero, so
    a subnormal positive operand keeps its value and yields the finite
    logarithm near ``-744`` that it has.
    """
    if values.dtype.kind in "iu":
        values = values.astype(numpy.float64, copy=False)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = numpy.asarray(values, dtype=numpy.float64)
        result = numpy.log(working)
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Log kernel returned an unexpected result size")
    return storage
