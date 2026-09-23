"""Python implementation of the cos VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def cos_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order cos VJP on prepared native values."""
    result = [
        -upstream * math.sin(float(item)) for upstream, item in zip(grad_values, values)
    ]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("cos VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
