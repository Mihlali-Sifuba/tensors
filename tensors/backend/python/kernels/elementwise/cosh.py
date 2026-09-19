"""Reference hyperbolic cosine for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf


def cosh(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the hyperbolic cosine of every element."""
    evaluate = _cosh
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
