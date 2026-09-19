"""Reference constant-filled construction for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape

if TYPE_CHECKING:
    from tensors.dtype import DataType


def full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage:
    """Return storage of ``shape`` holding one repeated value."""
    return PythonStorage.from_values(
        [fill_value] * Shape.from_iterable(shape).size,
        dtype,
    )
