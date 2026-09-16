"""Dispatch for reductions, extrema indices, and shape summation."""

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


def execute_sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage:
    """Run a fused accelerated multiply-and-broadcast reduction when safe."""
    from tensors.backend.python.kernels.reductions.sum_products_to_shape import (
        sum_products_to_shape as reference,
    )

    work = max(gradient.size, factor.size)
    if not _array_work_is_large_enough(work, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(gradient, factor, shape)
    sum_products_to_shape = _backend_kernel("sum_products_to_shape")
    result = sum_products_to_shape(gradient, factor, shape)
    if result is not None:
        return result
    return reference(gradient, factor, shape)
