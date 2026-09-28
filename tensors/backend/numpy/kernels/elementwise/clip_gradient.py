"""NumPy implementation of the clip VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def clip_gradient(
    grad_values: Any,
    values: Any,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Pass gradients strictly inside the bounds and zero the boundaries."""
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        mask = numpy.ones(output_shape, dtype=bool)
        if min_value is not None:
            mask &= values > min_value
        if max_value is not None:
            mask &= values < max_value
        result = numpy.where(
            numpy.isnan(values), numpy.nan, numpy.where(mask, grad_values, 0.0)
        )
        narrowed = result.astype(numpy.dtype(dtype.name), copy=False)
    storage = NumPyStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("clip VJP kernel returned an unexpected result size")
    return storage
