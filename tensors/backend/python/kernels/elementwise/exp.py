"""Reference the exponential for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _exp(value):
    try:
        return _math.exp(float(value))
    except OverflowError:
        return _math.inf


def exp(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the exponential of every element."""
    evaluate = _exp
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
