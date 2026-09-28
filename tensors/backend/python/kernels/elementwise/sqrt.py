"""Reference square root for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math as _math


def sqrt(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the correctly rounded square root of every prepared value.

    ``float`` performs the specified integer conversion explicitly: an
    integer operand becomes binary64 first, and the square root is taken of
    that converted value rather than of the mathematical integer.

    A negative operand is a value, not an error, and yields NaN.
    ``math.sqrt`` already returns ``-0.0`` for ``-0.0`` and NaN for NaN, and
    neither compares less than zero, so only a genuinely negative operand
    takes the NaN branch. A ``float32`` result is rounded once more by the
    typed buffer, which is exact for square root: binary64 carries more than
    the ``2p + 2`` bits that makes the second rounding agree with rounding
    the true root directly.
    """
    result = []
    for item in values:
        converted = float(item)
        result.append(_math.nan if converted < 0.0 else _math.sqrt(converted))
    if len(result) != _math.prod(output_shape):
        raise RuntimeError("Sqrt kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
