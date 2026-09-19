"""Dispatch for normalization, probability, and loss kernels."""

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


def execute_log_softmax_gradient(grad: Tensor, value: Tensor, axis: int) -> Storage:
    """Run a fused softmax-family VJP when cancellation risk is low."""
    from tensors.backend.python.kernels.nn.log_softmax_gradient import (
        log_softmax_gradient as reference,
    )

    if not _array_work_is_large_enough(
        max(grad.size, value.size), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(grad, value, axis)
    normalization_gradient = _backend_kernel("log_softmax_gradient")
    result = normalization_gradient(grad, value, axis)
    if result is not None:
        return result
    return reference(grad, value, axis)
