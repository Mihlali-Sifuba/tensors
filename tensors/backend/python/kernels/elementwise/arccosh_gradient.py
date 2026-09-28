"""Python implementation of the arccosh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arccosh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arccosh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        working = float(item)
        if math.isinf(working):
            derivative = 0.0
        else:
            derivative = 1.0 / (math.sqrt(working - 1.0) * math.sqrt(working + 1.0))
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arccosh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
