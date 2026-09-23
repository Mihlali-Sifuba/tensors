"""NumPy implementation of the minimum VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def minimum_gradient(
    grad_values: Any,
    left_values: Any,
    right_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
    reject_nondifferentiable: bool = False,
) -> tuple[Storage | None, Storage | None]:
    """Route gradients to smaller values and split exact ties equally."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape
        for values in (grad_values, left_values, right_values)
    ):
        raise RuntimeError(
            "minimum_gradient kernel received operands not prepared for output_shape"
        )
    need_left, need_right = needs_input_grad
    has_nan = numpy.isnan(left_values) | numpy.isnan(right_values)
    ties = left_values == right_values
    if reject_nondifferentiable:
        if bool(numpy.any(has_nan)):
            raise ValueError(
                "Higher-order derivatives of elementwise extrema are undefined at NaN"
            )
        if bool(numpy.any(ties)):
            raise ValueError(
                "Higher-order derivatives of elementwise extrema are undefined at ties"
            )
    left_selected = left_values < right_values
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
        raise RuntimeError("minimum VJP kernel returned an unexpected result size")
    return left, right
