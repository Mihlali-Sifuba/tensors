"""NumPy implementation of constant-filled construction."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def full(
    shape: tuple[int, ...], fill_value: int | float, *, dtype: DataType
) -> Storage | None:
    """Create constant-filled canonical storage."""
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.full(shape, fill_value, dtype=working_dtype)
    return _storage(result, dtype=dtype, output_shape=shape)
