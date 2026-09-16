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


def execute_power_exponent_gradient(
    grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage:
    """Run the accelerated power-exponent VJP when numerically safe."""
    from tensors.backend.python.kernels.elementwise.power_exponent_gradient import (
        power_exponent_gradient as reference,
    )

    if not _array_work_is_large_enough(
        max(grad.size, base.size, exponent.size), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(grad, base, exponent)
    power_exponent_gradient = _backend_kernel("power_exponent_gradient")
    result = power_exponent_gradient(grad, base, exponent)
    if result is not None:
        return result
    return reference(grad, base, exponent)
