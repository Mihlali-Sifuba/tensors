"""CuPy implementation of evenly spaced construction."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def linspace(
    start: int | float, stop: int | float, count: int, *, dtype: DataType
) -> Storage | None:
    """Create evenly spaced values when ordinary vector arithmetic is safe."""
    if count < 2:
        return None
    start_value = float(start)
    stop_value = float(stop)
    if start_value * stop_value < 0.0:
        return None
    limit = cupy.finfo(cupy.float64).max / 4.0
    if abs(start_value) > limit or abs(stop_value) > limit:
        return None
    fractions = cupy.arange(count, dtype=cupy.float64) / (count - 1)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = start_value * (1.0 - fractions) + stop_value * fractions
    result[0] = start
    result[-1] = stop
    return _storage(result, dtype=dtype, output_shape=(count,))
