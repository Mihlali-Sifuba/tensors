"""Reference the rectified linear unit for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math


def _relu(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value if value > 0 else 0


def relu(value: Tensor, *, dtype: DataType) -> Storage:
    """Rectify elementwise, leaving NaN in place."""
    evaluate = _relu
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
