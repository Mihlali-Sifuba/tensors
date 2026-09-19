"""Reference the sign function for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _sign(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def sign(value: Tensor, *, dtype: DataType) -> Storage:
    """Return -1, 0, or 1 for every element."""
    evaluate = _sign
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
