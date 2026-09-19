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


def execute_sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage:
    """Reduce broadcast-gradient contributions with an accelerated backend."""
    from tensors.backend.python.kernels.reductions.sum_to_shape import (
        sum_to_shape as reference,
    )

    if not _array_work_is_large_enough(gradient.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(gradient, shape)
    sum_to_shape = _backend_kernel("sum_to_shape")
    result = sum_to_shape(gradient, shape)
    if result is not None:
        return result
    return reference(gradient, shape)
