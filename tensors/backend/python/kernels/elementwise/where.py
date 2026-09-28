"""Python implementation of where."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def where(
    condition_values: Iterable[int | float],
    left_values: Iterable[int | float],
    right_values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select from prepared value pairs using a prepared condition mask."""
    result = [
        left if condition != 0 else right
        for condition, left, right in zip(condition_values, left_values, right_values)
    ]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("where kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
