"""Reference identity-like matrix construction for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def eye(rows: int, columns: int, k: int, *, dtype: DataType) -> Storage:
    """Return a matrix with ones on diagonal ``k`` and zeros elsewhere."""
    return PythonStorage.from_values(
        [
            1 if column - row == k else 0
            for row in range(rows)
            for column in range(columns)
        ],
        dtype,
    )
