"""Python implementation of the cosh VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def cosh_gradient(
    grad_values: Iterable[int | float],
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order cosh VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        try:
            derivative = math.sinh(float(item))
        except OverflowError:
            derivative = math.copysign(math.inf, item)
        result.append(upstream * derivative)
    if len(result) != math.prod(output_shape):
        raise RuntimeError("cosh VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
