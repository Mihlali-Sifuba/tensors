"""Python implementation of maximum."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def maximum(
    left_values: Iterable[int | float],
    right_values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select larger values, propagating NaN and choosing the left tie."""
    result = []
    for left, right in zip(left_values, right_values):
        if isinstance(left, float) and math.isnan(left):
            result.append(left)
        elif isinstance(right, float) and math.isnan(right):
            result.append(right)
        else:
            result.append(left if left >= right else right)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("maximum kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
