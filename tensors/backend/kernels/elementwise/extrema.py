"""Pairwise elementwise minimum and maximum, and their VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _numpy, _storage, _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor
    from ...types import ExtremumOperation

def extremum(
    operation: ExtremumOperation,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise minimum or maximum."""
    numpy = _numpy()
    function = numpy.minimum if operation == "minimum" else numpy.maximum
    try:
        result = function(_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def extremum_gradient(
    operation: ExtremumOperation,
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split an elementwise-extremum VJP for the requested operands."""
    numpy = _numpy()
    need_left, need_right = needs_input_grad
    try:
        left_values, right_values = numpy.broadcast_arrays(
            _view(left, numpy),
            _view(right, numpy),
        )
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    has_nan = numpy.isnan(left_values) | numpy.isnan(right_values)
    ties = left_values == right_values
    left_selected = (
        left_values > right_values
        if operation == "maximum"
        else left_values < right_values
    )
    left_storage = None
    if need_left:
        left_weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 1.0, 0.0)),
        )
        left_storage = _storage(
            upstream * left_weight,
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if left_storage is None:
            return None
    right_storage = None
    if need_right:
        right_weight = numpy.where(
            has_nan,
            numpy.nan,
            numpy.where(ties, 0.5, numpy.where(left_selected, 0.0, 1.0)),
        )
        right_storage = _storage(
            upstream * right_weight,
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if right_storage is None:
            return None
    return left_storage, right_storage
