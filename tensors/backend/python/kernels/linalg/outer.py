"""Reference vector outer products evaluated with Python arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend.python.storage import PythonStorage

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer(
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Multiply every pair of vector elements."""
    values = [left * right for left in left_values for right in right_values]
    return PythonStorage.from_values(values, dtype)
