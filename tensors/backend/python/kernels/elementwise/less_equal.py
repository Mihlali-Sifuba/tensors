"""Python implementation of less_equal."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import uint8


def less_equal(
    left_values: Iterable[int | float],
    right_values: Iterable[int | float],
    *,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the broadcasting less_equal comparison on prepared values."""
    result = [int(left <= right) for left, right in zip(left_values, right_values)]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("less_equal kernel returned an unexpected result size")
    return PythonStorage.from_values(result, uint8)
