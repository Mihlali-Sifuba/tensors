"""Python implementation of the arcsinh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arcsinh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arcsinh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        working = float(item)
        magnitude = abs(working)
        if math.isinf(magnitude):
            derivative = 0.0
        elif magnitude <= 1.0:
            derivative = 1.0 / math.sqrt(1.0 + working * working)
        else:
            reciprocal = 1.0 / magnitude
            derivative = reciprocal / math.sqrt(1.0 + reciprocal * reciprocal)
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arcsinh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
