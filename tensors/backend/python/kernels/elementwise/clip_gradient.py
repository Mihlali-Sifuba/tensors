"""Python implementation of the clip VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def clip_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Pass gradients strictly inside the bounds and zero the boundaries."""
    result = []
    for upstream, item in zip(grad_values, values):
        if isinstance(item, float) and math.isnan(item):
            result.append(math.nan)
            continue
        above_minimum = min_value is None or item > min_value
        below_maximum = max_value is None or item < max_value
        result.append(upstream if above_minimum and below_maximum else 0.0)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("clip VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
