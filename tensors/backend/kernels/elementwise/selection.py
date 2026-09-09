"""Predicate-driven elementwise selection and its VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _numpy, _storage, _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run broadcasting elementwise selection."""
    numpy = _numpy()
    try:
        result = numpy.where(
            _view(condition, numpy) != 0,
            _view(left, numpy),
            _view(right, numpy),
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def where_gradient(
    grad: Tensor,
    condition: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split a selection VJP into the requested left and right terms."""
    numpy = _numpy()
    need_left, need_right = needs_input_grad
    try:
        selected = numpy.broadcast_to(_view(condition, numpy), grad.shape) != 0
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left = None
    if need_left:
        left = _storage(
            numpy.where(selected, upstream, 0.0),
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if left is None:
            return None
    right = None
    if need_right:
        right = _storage(
            numpy.where(selected, 0.0, upstream),
            dtype=grad.dtype,
            output_shape=grad.shape,
            numpy=numpy,
        )
        if right is None:
            return None
    return left, right
