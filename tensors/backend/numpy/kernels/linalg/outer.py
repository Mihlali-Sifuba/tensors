"""NumPy implementation of vector outer products."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType


def outer(
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Execute a floating outer product with NumPy-native values."""
    if dtype.kind != "floating":
        return None
    try:
        left = left_values.astype(numpy.float64, copy=False)
        right = right_values.astype(numpy.float64, copy=False)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = numpy.multiply.outer(left, right)
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
