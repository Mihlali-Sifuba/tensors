"""Reference square root for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _sqrt(value):
    if value < 0:
        raise ValueError("sqrt is only defined for non-negative values")
    return _math.sqrt(float(value))


def sqrt(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the square root of every element."""
    evaluate = _sqrt
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
