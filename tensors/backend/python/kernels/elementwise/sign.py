"""Reference the sign function for the Python backend."""

from __future__ import annotations
from collections.abc import Iterable
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math


def _sign(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def sign(
    values: Iterable[int | float],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Classify each prepared value as -1, 0 or 1.

    Both signed zeros compare equal to zero, so each returns the integer 0
    and the declared dtype renders it as canonical positive zero.
    """
    result = [_sign(item) for item in values]
    if len(result) != math.prod(output_shape):
        raise RuntimeError("Sign kernel returned an unexpected result size")
    return PythonStorage.from_values(result, dtype)
