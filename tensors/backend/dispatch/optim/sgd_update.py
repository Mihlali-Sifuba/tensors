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


def execute_sgd_update(
    parameter: Tensor, gradient: Tensor, learning_rate: float
) -> Storage:
    """Run a fused SGD parameter update."""
    from tensors.backend.python.kernels.optim.sgd_update import sgd_update as reference

    if not _array_work_is_large_enough(parameter.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(parameter, gradient, learning_rate)
    sgd_update = _backend_kernel("sgd_update")
    result = sgd_update(parameter, gradient, learning_rate)
    if result is not None:
        return result
    return reference(parameter, gradient, learning_rate)
