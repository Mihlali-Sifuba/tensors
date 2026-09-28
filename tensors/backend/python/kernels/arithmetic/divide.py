"""Reference division for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
import math
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


_INFINITY = float("inf")
_NAN = float("nan")


def divide(
    left: Iterable[int | float],
    right: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Divide each prepared pair with Python scalar semantics."""
    def evaluate(x, y):
        if y == 0:
            # Python raises where IEEE 754 delivers a value. Integer operands
            # never reach here: the operation layer rejects a zero integer
            # denominator, because there is no integer infinity to return.
            # See docs/arithmetic-semantics.md section 7.2.
            if x == 0 or x != x:
                return _NAN
            return math.copysign(
                _INFINITY, math.copysign(1.0, x) * math.copysign(1.0, y)
            )
        return x / y

    values = [evaluate(x, y) for x, y in zip(left, right)]
    return PythonStorage.from_arithmetic(values, dtype)
