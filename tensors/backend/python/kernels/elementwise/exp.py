"""Reference the exponential for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType

_INFINITY = float("inf")


def exp(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the exponential of every prepared value.

    ``math.exp`` raises ``OverflowError`` where IEEE 754 delivers ``+inf``,
    so the overflow is translated rather than propagated: ``exp`` has a
    result for every input and never fails on a value. It underflows to zero
    silently, which needs no translation, and the subnormal band below about
    ``x = -708`` in binary64 is returned as it stands.

    An integer operand is converted to binary64 explicitly, which is the
    specified conversion rather than a promise about the mathematical
    integer.
    """
    result = []
    for item in values:
        try:
            result.append(math.exp(float(item)))
        except OverflowError:
            result.append(_INFINITY)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Exp kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
