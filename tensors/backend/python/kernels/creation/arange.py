"""Reference arithmetic-progression construction for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage:
    """Return ``count`` values starting at ``start`` and spaced by ``step``."""
    return PythonStorage.from_values(
        [start + index * step for index in range(count)],
        dtype,
    )
