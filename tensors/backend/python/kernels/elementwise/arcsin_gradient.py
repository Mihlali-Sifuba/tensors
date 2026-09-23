"""Python implementation of the arcsin VJP."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arcsin_gradient(
    grad_values: Iterable[float],
    values: Iterable[float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate the first-order arcsin VJP on prepared native values."""
    result = [
        upstream / math.sqrt(1.0 - float(item) ** 2.0)
        for upstream, item in zip(grad_values, values)
    ]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arcsin VJP kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
