"""Python implementation of the sinh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def sinh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order sinh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        try:
            derivative = math.cosh(float(item))
        except OverflowError:
            derivative = math.inf
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("sinh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
