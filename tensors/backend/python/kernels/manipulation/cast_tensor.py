"""Reference dtype conversion for Python storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.casting import cast_values

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage:
    """Convert every element with Python scalar conversion semantics."""
    values = cast_values(value._data, source_dtype=value.dtype, target_dtype=dtype)
    return PythonStorage.from_values(values, dtype)
