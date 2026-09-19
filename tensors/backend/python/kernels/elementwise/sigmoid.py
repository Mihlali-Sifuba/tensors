"""Reference the logistic function for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)


def sigmoid(value: Tensor, *, dtype: DataType) -> Storage:
    """Apply the logistic function from whichever side avoids overflow."""
    evaluate = _sigmoid
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
