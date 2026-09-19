"""Reference the vector outer product for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def outer(left: Tensor, right: Tensor, *, dtype: DataType) -> Storage | None:
    """Multiply every pair of elements from two vectors."""
    a = left
    b = right
    values = [left * right for left in a._data for right in b._data]
    return PythonStorage.from_values(values, dtype)
