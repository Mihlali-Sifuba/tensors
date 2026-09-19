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


def execute_exp_gradient(grad: Tensor, value: Tensor) -> Storage:
    """Run an accelerated unary VJP, or the Python reference."""
    from tensors.backend.python.kernels.elementwise.exp_gradient import (
        exp_gradient as reference,
    )

    if not _array_work_is_large_enough(
        max(grad.size, value.size), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(grad, value)
    unary_gradient = _backend_kernel("exp_gradient")
    result = unary_gradient(grad, value)
    if result is not None:
        return result
    return reference(grad, value)
