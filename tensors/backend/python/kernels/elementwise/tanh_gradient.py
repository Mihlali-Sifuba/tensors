"""Python implementation of the tanh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def tanh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order tanh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        magnitude = math.exp(-2.0 * abs(float(item)))
        denominator = 1.0 + magnitude
        result.append(upstream * (4.0 * magnitude / (denominator * denominator)))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("tanh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
