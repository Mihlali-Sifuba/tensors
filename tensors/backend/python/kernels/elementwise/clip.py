"""Reference clipping for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Bound every element to the requested interval."""
    values = []
    for item in value._data:
        if min_value is not None and item < min_value:
            item = min_value
        if max_value is not None and item > max_value:
            item = max_value
        values.append(item)
    return PythonStorage.from_values(values, dtype)
