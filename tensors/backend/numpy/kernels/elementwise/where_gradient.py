"""NumPy implementation of the where VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def where_gradient(
    grad_values: Any,
    condition_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Route native upstream values into the requested data branches."""
    expected_shape = tuple(output_shape)
    if any(
        tuple(values.shape) != expected_shape
        for values in (grad_values, condition_values)
    ):
        raise RuntimeError(
            "where_gradient kernel received operands not prepared for output_shape"
        )
    need_left, need_right = needs_input_grad
    selected = condition_values != 0
    left = None
    if need_left:
        left = NumPyStorage(
            numpy.where(selected, grad_values, 0.0).astype(
                numpy.dtype(dtype.name), copy=False
            ),
            dtype,
        )
    right = None
    if need_right:
        right = NumPyStorage(
            numpy.where(selected, 0.0, grad_values).astype(
                numpy.dtype(dtype.name), copy=False
            ),
            dtype,
        )
    expected = math.prod(output_shape)
    if (left is not None and left.size != expected) or (
        right is not None and right.size != expected
    ):
        raise RuntimeError("where VJP kernel returned an unexpected result size")
    return left, right
