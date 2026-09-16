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


def execute_rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage]:
    """Run a fused RMSprop update on ordinary finite optimizer state."""
    from tensors.backend.python.kernels.optim.rmsprop_update import (
        rmsprop_update as reference,
    )

    if not _array_work_is_large_enough(parameter.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            parameter,
            gradient,
            scale,
            scaled,
            rho=rho,
            learning_rate=learning_rate,
            epsilon=epsilon,
        )
    rmsprop_update = _backend_kernel("rmsprop_update")
    result = rmsprop_update(
        parameter,
        gradient,
        scale,
        scaled,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
    if result is not None:
        return result
    return reference(
        parameter,
        gradient,
        scale,
        scaled,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
