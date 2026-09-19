"""Dispatch for matrix and vector products and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import _NUMPY_MATMUL_MIN_WORK, _array_work_is_large_enough
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run outer-product VJPs when native sums preserve semantics."""
    from tensors.backend.python.kernels.linalg.outer_gradient import (
        outer_gradient as reference,
    )

    if not _array_work_is_large_enough(grad.size, _NUMPY_MATMUL_MIN_WORK):
        return reference(grad, left, right, needs_input_grad=needs_input_grad)
    outer_gradient = _backend_kernel("outer_gradient")
    result = outer_gradient(grad, left, right, needs_input_grad=needs_input_grad)
    if result is not None:
        return result
    return reference(grad, left, right, needs_input_grad=needs_input_grad)
