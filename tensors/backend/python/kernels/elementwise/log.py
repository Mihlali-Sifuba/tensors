"""Reference the natural logarithm for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _log(value):
    if value <= 0:
        raise ValueError("log is only defined for positive values")
    return _math.log(float(value))


def log(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the natural logarithm of every element."""
    evaluate = _log
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
