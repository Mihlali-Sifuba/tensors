"""Python implementation of cosh."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def cosh(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate cosh on prepared native values."""
    result = []
    for item in values:
        try:
            result.append(math.cosh(float(item)))
        except OverflowError:
            result.append(math.inf)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("cosh kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
