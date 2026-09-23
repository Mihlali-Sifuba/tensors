"""NumPy implementation of maximum."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def maximum(
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select larger values, propagating NaN and choosing the left tie."""
    result = numpy.where(
        numpy.isnan(left_values),
        left_values,
        numpy.where(
            numpy.isnan(right_values),
            right_values,
            numpy.where(left_values >= right_values, left_values, right_values),
        ),
    )
    narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("maximum kernel returned an unexpected result size")
    return storage
