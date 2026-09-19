"""NumPy implementation of the vector outer product VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view
from tensors.backend.numpy.kernels.reductions.stability import _stable_sum_candidate

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested stable native outer-product VJPs."""
    upstream = _view(grad).astype(numpy.float64, copy=False)
    left_values = _view(left).astype(numpy.float64, copy=False)
    right_values = _view(right).astype(numpy.float64, copy=False)
    if not _finite_operands(upstream, left_values, right_values):
        return None
    need_left, need_right = needs_input_grad
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        left_terms = upstream * right_values if need_left else None
        right_terms = upstream * left_values[:, None] if need_right else None
    if left_terms is not None and (not _stable_sum_candidate(left_terms, (1,))):
        return None
    if right_terms is not None and (not _stable_sum_candidate(right_terms, (0,))):
        return None
    left_storage = None
    if left_terms is not None:
        left_storage = _storage(
            numpy.sum(left_terms, axis=1), dtype=grad.dtype, output_shape=left.shape
        )
        if left_storage is None:
            return None
    right_storage = None
    if right_terms is not None:
        right_storage = _storage(
            numpy.sum(right_terms, axis=0), dtype=grad.dtype, output_shape=right.shape
        )
        if right_storage is None:
            return None
    return (left_storage, right_storage)
