"""NumPy implementation of identity-like matrix construction."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def eye(rows: int, columns: int, k: int, *, dtype: DataType) -> Storage | None:
    """Create identity-like canonical storage."""
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.eye(rows, columns, k=k, dtype=working_dtype)
    return _storage(result, dtype=dtype, output_shape=(rows, columns))
