"""CuPy implementation of the elementwise minimum VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import tensor_to_logical_array
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def minimum_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split an elementwise-extremum VJP for the requested operands."""
    need_left, need_right = needs_input_grad
    try:
        left_values, right_values = cupy.broadcast_arrays(
            tensor_to_logical_array(left), tensor_to_logical_array(right)
        )
    except ValueError:
        return None
    upstream = _working_values(grad)
    has_nan = cupy.isnan(left_values) | cupy.isnan(right_values)
    ties = left_values == right_values
    left_selected = left_values < right_values
    left_storage = None
    if need_left:
        left_weight = cupy.where(
            has_nan,
            cupy.nan,
            cupy.where(ties, 0.5, cupy.where(left_selected, 1.0, 0.0)),
        )
        left_storage = _storage(
            upstream * left_weight, dtype=grad.dtype, output_shape=grad.shape
        )
        if left_storage is None:
            return None
    right_storage = None
    if need_right:
        right_weight = cupy.where(
            has_nan,
            cupy.nan,
            cupy.where(ties, 0.5, cupy.where(left_selected, 0.0, 1.0)),
        )
        right_storage = _storage(
            upstream * right_weight, dtype=grad.dtype, output_shape=grad.shape
        )
        if right_storage is None:
            return None
    return (left_storage, right_storage)
