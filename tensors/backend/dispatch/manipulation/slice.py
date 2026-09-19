"""Dispatch for shape, layout, indexing, and representation changes."""

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
    from tensors._typing import TensorIndex
    from tensors.tensor import Tensor


def execute_slice(
    value: Tensor, key: TensorIndex, *, output_shape: tuple[int, ...]
) -> Storage:
    """Run accelerated tensor slicing, or the Python reference."""
    from tensors.backend.python.kernels.manipulation.slice_tensor import (
        slice_tensor as reference,
    )

    if not _array_work_is_large_enough(
        _shape_size(output_shape), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(value, key, output_shape=output_shape)
    slice_tensor = _backend_kernel("slice_tensor")
    result = slice_tensor(value, key, output_shape=output_shape)
    if result is not None:
        return result
    return reference(value, key, output_shape=output_shape)
