"""Reference arcsine for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _arcsin(value):
    if value < -1.0 or value > 1.0:
        raise ValueError("arcsin is only defined for values between -1 and 1")
    return math.asin(float(value))


def arcsin(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the arcsine of every element, in radians."""
    evaluate = _arcsin
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
