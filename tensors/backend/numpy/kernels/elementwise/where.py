"""NumPy implementation of where."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def where(
    condition_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Select between native data arrays using a native condition array."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape
        for values in (condition_values, left_values, right_values)
    ):
        raise RuntimeError(
            "where kernel received operands not prepared for output_shape"
        )
    result = numpy.where(condition_values != 0, left_values, right_values)
    narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("where kernel returned an unexpected result size")
    return storage
