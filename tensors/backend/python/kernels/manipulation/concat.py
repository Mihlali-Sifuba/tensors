"""Copy inputs along an existing axis using Python-native storage."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def concat(
    values: Sequence[Any],
    shapes: Sequence[tuple[int, ...]],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> PythonStorage:
    """Copy compact native buffers along an existing axis."""
    if not shapes[0]:
        return PythonStorage.from_values((value[0] for value in values), dtype)
    trailing = math.prod(output_shape[axis + 1 :])
    groups = math.prod(output_shape[:axis])
    result = []
    for group in range(groups):
        for value, shape in zip(values, shapes):
            count = shape[axis] * trailing
            start = group * count
            result.extend(value[start : start + count])
    return PythonStorage.from_values(result, dtype)
