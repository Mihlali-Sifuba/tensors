"""Dispatch for optimizer updates."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_rmsprop_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible RMSprop states in one native array batch."""
    count = len(parameters)
    if (
        count < 2
        or any((len(items) != count for items in (gradients, scales, scaled_values)))
        or (
            not _array_work_is_large_enough(
                sum((parameter.size for parameter in parameters)),
                _NUMPY_ELEMENTWISE_MIN_SIZE,
            )
        )
    ):
        return None
    rmsprop_updates = _backend_kernel("rmsprop_updates")
    return rmsprop_updates(
        parameters,
        gradients,
        scales,
        scaled_values,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
