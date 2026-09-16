"""Dispatch for normalization, probability, and loss kernels."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage
from tensors.backend.types import LossReduction

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run the requested binary cross-entropy VJPs."""
    from tensors.backend.python.kernels.nn.binary_cross_entropy_gradient import (
        binary_cross_entropy_gradient as reference,
    )

    if not _array_work_is_large_enough(prediction.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            grad,
            prediction,
            target,
            from_logits=from_logits,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
    binary_cross_entropy_gradient = _backend_kernel("binary_cross_entropy_gradient")
    result = binary_cross_entropy_gradient(
        grad,
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
        needs_input_grad=needs_input_grad,
    )
    if result is not None:
        return result
    return reference(
        grad,
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
        needs_input_grad=needs_input_grad,
    )
