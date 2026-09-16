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


def execute_minimum_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Split an elementwise-extremum VJP, including tie sharing."""
    from tensors.backend.python.kernels.elementwise.minimum_gradient import (
        minimum_gradient as reference,
    )

    if not _array_work_is_large_enough(grad.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(grad, left, right, needs_input_grad=needs_input_grad)
    extremum_gradient = _backend_kernel("minimum_gradient")
    result = extremum_gradient(grad, left, right, needs_input_grad=needs_input_grad)
    if result is not None:
        return result
    return reference(grad, left, right, needs_input_grad=needs_input_grad)
