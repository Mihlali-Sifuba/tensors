"""Python implementation of the arctanh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arctanh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arctanh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        working = float(item)
        result.append(upstream / (1.0 - working * working))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arctanh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
