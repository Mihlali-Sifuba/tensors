"""Reference inverse hyperbolic cosine for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _arccosh(value):
    if value < 1.0:
        raise ValueError(
            "arccosh is only defined for values greater than or equal to 1"
        )
    return math.acosh(float(value))


def arccosh(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the inverse hyperbolic cosine of every element."""
    evaluate = _arccosh
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
