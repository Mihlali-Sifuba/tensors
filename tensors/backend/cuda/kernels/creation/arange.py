"""CuPy implementation of arithmetic-progression construction."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def arange(
    start: int | float, step: int | float, count: int, *, dtype: DataType
) -> Storage | None:
    """Create an arithmetic progression from a validated element count."""
    integer_inputs = isinstance(start, int) and isinstance(step, int)
    if dtype.kind == "integer" or integer_inputs:
        return None
    working_dtype = object if integer_inputs else cupy.float64
    indices = cupy.arange(count, dtype=working_dtype)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = start + indices * step
    return _storage(result, dtype=dtype, output_shape=(count,))
