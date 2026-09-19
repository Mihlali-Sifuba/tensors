"""CuPy implementation of identity-like matrix construction."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def eye(rows: int, columns: int, k: int, *, dtype: DataType) -> Storage | None:
    """Create identity-like canonical storage."""
    if dtype.kind == "integer":
        return None
    working_dtype = object if dtype.kind == "integer" else cupy.float64
    result = cupy.eye(rows, columns, k=k, dtype=working_dtype)
    return _storage(result, dtype=dtype, output_shape=(rows, columns))
