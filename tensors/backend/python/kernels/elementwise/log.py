"""Reference the natural logarithm for the Python backend."""

from __future__ import annotations
import math
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def log(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the natural logarithm of every prepared value.

    The domain is settled before this runs: the dispatcher refuses a
    non-positive operand, so nothing here can reach ``math.log``'s own
    domain error. What remains is arithmetic. NaN is not refused, because it
    is not outside the domain so much as unknown, and ``math.log`` returns
    NaN for it; ``+inf`` returns ``+inf``; and a subnormal positive operand
    has an ordinary finite logarithm near ``-744``.

    An integer operand is converted to binary64 explicitly, which is the
    specified conversion rather than a promise about the mathematical
    integer.
    """
    result = [math.log(float(item)) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Log kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
