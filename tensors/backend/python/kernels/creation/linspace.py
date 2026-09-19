"""Reference evenly spaced construction for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage:
    """Return ``count`` values from ``start`` to ``stop``, both included.

    Interior points are interpolated rather than accumulated, so a long
    sequence does not drift, and both endpoints are stored exactly.
    """
    values: list[int | float]
    if count == 0:
        values = []
    elif count == 1:
        values = [start]
    else:
        intervals = count - 1
        values = [start]
        for index in range(1, intervals):
            fraction = index / intervals
            values.append(
                math.fsum((float(start) * (1.0 - fraction), float(stop) * fraction))
            )
        values.append(stop)
    return PythonStorage.from_values(values, dtype)
