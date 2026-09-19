"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_REDUCTION_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_reduce_variance_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage:
    """Run a fused reduction VJP, or the Python reference."""
    from tensors.backend.python.kernels.reductions.reduce_variance_gradient import (
        reduce_variance_gradient as reference,
    )

    if not _array_work_is_large_enough(value.size, _NUMPY_REDUCTION_MIN_SIZE):
        return reference(grad, value, axes, keepdims=keepdims)
    reduction_gradient = _backend_kernel("reduce_variance_gradient")
    result = reduction_gradient(grad, value, axes, keepdims=keepdims)
    if result is not None:
        return result
    return reference(grad, value, axes, keepdims=keepdims)
