"""Python implementation of not_equal."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import uint8


def not_equal(
    left_values: Iterable[int | float],
    right_values: Iterable[int | float],
    *,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the broadcasting not_equal comparison on prepared values."""
    result = [int(left != right) for left, right in zip(left_values, right_values)]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("not_equal kernel returned an unexpected result size")
    return PythonStorage.from_values(result, uint8)
