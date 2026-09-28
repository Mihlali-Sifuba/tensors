"""Python implementation of arcsinh."""

from __future__ import annotations

import math
from collections.abc import Iterable

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def arcsinh(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Evaluate arcsinh on prepared native values."""
    result = [math.asinh(float(item)) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("arcsinh kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
