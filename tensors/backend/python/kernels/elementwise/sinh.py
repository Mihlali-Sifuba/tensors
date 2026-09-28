"""Python implementation of sinh."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def sinh(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate sinh on prepared native values."""
    result = []
    for item in values:
        try:
            result.append(math.sinh(float(item)))
        except OverflowError:
            result.append(math.copysign(math.inf, item))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("sinh kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
