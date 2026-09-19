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


def execute_sgd_updates(
    parameters: Sequence[Tensor], gradients: Sequence[Tensor], learning_rate: float
) -> tuple[Storage, ...] | None:
    """Update several compatible parameters with one native array batch."""
    if (
        len(parameters) < 2
        or len(parameters) != len(gradients)
        or (
            not _array_work_is_large_enough(
                sum((parameter.size for parameter in parameters)),
                _NUMPY_ELEMENTWISE_MIN_SIZE,
            )
        )
    ):
        return None
    sgd_updates = _backend_kernel("sgd_updates")
    return sgd_updates(parameters, gradients, learning_rate)
