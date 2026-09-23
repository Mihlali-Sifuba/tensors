"""Python implementation of clip."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def clip(
    values: Iterable[int | float],
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Clip prepared values while preserving equality and NaN behavior."""
    result = []
    for item in values:
        if min_value is not None and item < min_value:
            item = min_value
        if max_value is not None and item > max_value:
            item = max_value
        result.append(item)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("clip kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
