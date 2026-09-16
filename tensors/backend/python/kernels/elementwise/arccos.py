"""Reference arccosine for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _arccos(value):
    if value < -1.0 or value > 1.0:
        raise ValueError("arccos is only defined for values between -1 and 1")
    return math.acos(float(value))


def arccos(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the arccosine of every element, in radians."""
    evaluate = _arccos
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
