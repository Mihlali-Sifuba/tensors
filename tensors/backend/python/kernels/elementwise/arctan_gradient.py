"""Python implementation of the arctan VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arctan_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arctan VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        item = float(item)
        if math.isinf(item):
            derivative = 0.0
        elif abs(item) <= 1.0:
            derivative = 1.0 / (1.0 + item * item)
        else:
            reciprocal = 1.0 / item
            square = reciprocal * reciprocal
            derivative = square / (1.0 + square)
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arctan VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
