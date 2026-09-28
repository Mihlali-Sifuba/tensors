"""Python implementation of the tan VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def tan_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order tan VJP on prepared native values."""
    result = []
    for upstream, item in zip(grad_values, values):
        cosine = math.cos(float(item))
        result.append(upstream / (cosine * cosine))
    if len(result) != math.prod(output_shape):
        raise RuntimeError("tan VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
