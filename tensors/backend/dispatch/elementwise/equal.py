"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
    _shape_size,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_equal(
    left: Tensor, right: Tensor, *, output_shape: tuple[int, ...]
) -> Storage:
    """Run an elementwise broadcasting comparison."""
    from tensors.backend.python.kernels.elementwise.equal import equal as reference

    if not _array_work_is_large_enough(
        _shape_size(output_shape), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(left, right, output_shape=output_shape)
    comparison = _backend_kernel("equal")
    result = comparison(left, right, output_shape=output_shape)
    if result is not None:
        return result
    return reference(left, right, output_shape=output_shape)
