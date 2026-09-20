"""Reference addition for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType


def add(
    left: Iterable[int | float],
    right: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Add each prepared pair with Python scalar semantics."""
    values = [x + y for x, y in zip(left, right)]
    return PythonStorage.from_arithmetic(values, dtype)
