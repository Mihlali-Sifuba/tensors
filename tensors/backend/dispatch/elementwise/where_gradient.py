"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_where_gradient(
    grad: Tensor,
    condition: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Split a selection gradient along a condition mask."""
    from tensors.backend.python.kernels.elementwise.where_gradient import (
        where_gradient as reference,
    )

    if not _array_work_is_large_enough(grad.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(grad, condition, needs_input_grad=needs_input_grad)
    where_gradient = _backend_kernel("where_gradient")
    result = where_gradient(grad, condition, needs_input_grad=needs_input_grad)
    if result is not None:
        return result
    return reference(grad, condition, needs_input_grad=needs_input_grad)
