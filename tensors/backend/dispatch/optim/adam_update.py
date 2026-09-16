"""Dispatch for optimizer updates."""

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


def execute_adam_update(
    parameter: Tensor,
    gradient: Tensor,
    moment: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_correction: float,
    second_correction: float,
) -> tuple[Storage, Storage, Storage, Storage, Storage]:
    """Run a fused Adam update on ordinary finite optimizer state."""
    from tensors.backend.python.kernels.optim.adam_update import (
        adam_update as reference,
    )

    if not _array_work_is_large_enough(parameter.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            parameter,
            gradient,
            moment,
            scale,
            scaled,
            beta1=beta1,
            beta2=beta2,
            learning_rate=learning_rate,
            epsilon=epsilon,
            first_correction=first_correction,
            second_correction=second_correction,
        )
    adam_update = _backend_kernel("adam_update")
    result = adam_update(
        parameter,
        gradient,
        moment,
        scale,
        scaled,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_correction=first_correction,
        second_correction=second_correction,
    )
    if result is not None:
        return result
    return reference(
        parameter,
        gradient,
        moment,
        scale,
        scaled,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_correction=first_correction,
        second_correction=second_correction,
    )
