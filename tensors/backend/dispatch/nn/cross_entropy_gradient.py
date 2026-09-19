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


def execute_cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Run the requested multiclass cross-entropy VJPs when safe."""
    from tensors.backend.python.kernels.nn.cross_entropy_gradient import (
        cross_entropy_gradient as reference,
    )

    if not _array_work_is_large_enough(logits.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            grad,
            logits,
            targets,
            axis,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
    cross_entropy_gradient = _backend_kernel("cross_entropy_gradient")
    result = cross_entropy_gradient(
        grad,
        logits,
        targets,
        axis,
        reduction=reduction,
        needs_input_grad=needs_input_grad,
    )
    if result is not None:
        return result
    return reference(
        grad,
        logits,
        targets,
        axis,
        reduction=reduction,
        needs_input_grad=needs_input_grad,
    )
