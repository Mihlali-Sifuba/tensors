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


def execute_division_denominator_gradient(
    grad: Tensor, numerator: Tensor, denominator: Tensor
) -> Storage:
    """Run the accelerated division-denominator VJP when numerically safe."""
    from tensors.backend.python.kernels.elementwise.division_denominator_gradient import (
        division_denominator_gradient as reference,
    )

    if not _array_work_is_large_enough(
        max(grad.size, numerator.size, denominator.size), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(grad, numerator, denominator)
    division_denominator_gradient = _backend_kernel("division_denominator_gradient")
    result = division_denominator_gradient(grad, numerator, denominator)
    if result is not None:
        return result
    return reference(grad, numerator, denominator)
