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


def execute_adam_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    moments: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_corrections: Sequence[float],
    second_corrections: Sequence[float],
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible Adam states in one native array batch."""
    count = len(parameters)
    if (
        count < 2
        or any(
            (
                len(items) != count
                for items in (
                    gradients,
                    moments,
                    scales,
                    scaled_values,
                    first_corrections,
                    second_corrections,
                )
            )
        )
        or (
            not _array_work_is_large_enough(
                sum((parameter.size for parameter in parameters)),
                _NUMPY_ELEMENTWISE_MIN_SIZE,
            )
        )
    ):
        return None
    adam_updates = _backend_kernel("adam_updates")
    return adam_updates(
        parameters,
        gradients,
        moments,
        scales,
        scaled_values,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_corrections=first_corrections,
        second_corrections=second_corrections,
    )
