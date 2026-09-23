"""NumPy implementation of the maximum VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def maximum_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Route gradients to larger values and split exact ties equally."""
    need_left, need_right = needs_input_grad
    left_values, right_values = numpy.broadcast_arrays(left_values, right_values)
    has_nan = numpy.isnan(left_values) | numpy.isnan(right_values)
    ties = left_values == right_values
    left_selected = left_values > right_values
    left = None
    if need_left:
        weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 1.0, 0.0)),
        )
        left = NumPyStorage(
            (grad_values * weight).astype(numpy.dtype(dtype.name), copy=False), dtype
        )
    right = None
    if need_right:
        weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 0.0, 1.0)),
        )
        right = NumPyStorage(
            (grad_values * weight).astype(numpy.dtype(dtype.name), copy=False), dtype
        )
    expected = math.prod(output_shape)
    if (left is not None and left.size != expected) or (
        right is not None and right.size != expected
    ):
        raise RuntimeError("maximum VJP kernel returned an unexpected result size")
    return left, right
