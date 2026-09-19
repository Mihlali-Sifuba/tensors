"""Reference softplus for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math as _math


def _softplus(value):
    value = float(value)
    return _math.log1p(_math.exp(-abs(value))) + max(value, 0.0)


def softplus(value: Tensor, *, dtype: DataType) -> Storage:
    """Apply softplus through log1p so a large input does not overflow."""
    evaluate = _softplus
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
