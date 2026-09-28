"""NumPy implementation of square root."""

from __future__ import annotations
import math
import numpy
from typing import TYPE_CHECKING, Any
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def sqrt(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype.

    An integer operand is converted to binary64 explicitly before the root is
    taken, which is the specified conversion rather than a promise about the
    mathematical integer. ``numpy.sqrt`` is correctly rounded in the format it
    is given, returns NaN for a negative operand rather than raising, and
    keeps ``-0.0``. Host arithmetic has no flush-to-zero, so a subnormal
    operand keeps its value.
    """
    if values.dtype.kind in "iu":
        values = values.astype(numpy.float64, copy=False)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = numpy.sqrt(values)
    storage = NumPyStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sqrt kernel returned an unexpected result size")
    return storage
