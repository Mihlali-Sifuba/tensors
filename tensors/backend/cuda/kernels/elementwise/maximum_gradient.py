"""CUDA implementation of the maximum VJP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
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
    upstream = _widen(grad_values)
    left_values = _widen(left_values)
    right_values = _widen(right_values)
    left_values, right_values = cupy.broadcast_arrays(left_values, right_values)
    has_nan = cupy.isnan(left_values) | cupy.isnan(right_values)
    ties = left_values == right_values
    left_selected = left_values > right_values
    left = None
    if need_left:
        weight = cupy.where(
            has_nan,
            cupy.nan,
            cupy.where(ties, 0.5, cupy.where(left_selected, 1.0, 0.0)),
        )
        left = CudaStorage(_narrow(upstream * weight, cupy.dtype(dtype.name)), dtype)
    right = None
    if need_right:
        weight = cupy.where(
            has_nan,
            cupy.nan,
            cupy.where(ties, 0.5, cupy.where(left_selected, 0.0, 1.0)),
        )
        right = CudaStorage(_narrow(upstream * weight, cupy.dtype(dtype.name)), dtype)
    expected = math.prod(output_shape)
    if (left is not None and left.size != expected) or (
        right is not None and right.size != expected
    ):
        raise RuntimeError("maximum VJP kernel returned an unexpected result size")
    return left, right
