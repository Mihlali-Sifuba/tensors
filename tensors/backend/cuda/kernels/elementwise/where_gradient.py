"""CuPy implementation of the elementwise selection VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def where_gradient(
    grad: Tensor,
    condition: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split a selection VJP into the requested left and right terms."""
    need_left, need_right = needs_input_grad
    try:
        selected = cupy.broadcast_to(_view(condition), grad.shape) != 0
    except ValueError:
        return None
    upstream = _view(grad).astype(cupy.float64, copy=False)
    left = None
    if need_left:
        left = _storage(
            cupy.where(selected, upstream, 0.0),
            dtype=grad.dtype,
            output_shape=grad.shape,
        )
        if left is None:
            return None
    right = None
    if need_right:
        right = _storage(
            cupy.where(selected, 0.0, upstream),
            dtype=grad.dtype,
            output_shape=grad.shape,
        )
        if right is None:
            return None
    return (left, right)
