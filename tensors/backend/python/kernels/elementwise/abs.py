"""Reference absolute value for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import builtins


def abs(value: Tensor, *, dtype: DataType) -> Storage:
    """Return the magnitude of every element."""
    evaluate = builtins.abs
    return PythonStorage.from_values([evaluate(item) for item in value._data], dtype)
