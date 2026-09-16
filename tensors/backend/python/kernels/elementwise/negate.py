"""Reference negation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Return the additive inverse of every element."""
    a = value
    data = [-x for x in a._data]
    return PythonStorage.from_values(data, dtype)
