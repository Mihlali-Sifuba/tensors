"""NumPy implementation of clip."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def clip(
    values: Any,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Clip native values while preserving equality, signed zero, and NaN."""
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = values
        if min_value is not None:
            result = numpy.where(result < min_value, min_value, result)
        if max_value is not None:
            result = numpy.where(result > max_value, max_value, result)
        narrowed = numpy.asarray(result).astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("clip kernel returned an unexpected result size")
    return storage
