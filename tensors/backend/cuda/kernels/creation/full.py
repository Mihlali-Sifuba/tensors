"""CuPy implementation of constant-filled construction."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def full(
    shape: tuple[int, ...], fill_value: int | float, *, dtype: DataType
) -> Storage | None:
    """Create constant-filled canonical storage."""
    if dtype.kind == "integer":
        return None
    working_dtype = object if dtype.kind == "integer" else cupy.float64
    result = cupy.full(shape, fill_value, dtype=working_dtype)
    return _storage(result, dtype=dtype, output_shape=shape)
