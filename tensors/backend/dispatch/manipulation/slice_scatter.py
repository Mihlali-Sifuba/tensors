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
    from tensors.tensor import Tensor


def execute_slice_scatter(
    value: Tensor, indices: list[int], *, output_shape: tuple[int, ...]
) -> Storage:
    """Run accelerated slice scattering, or the Python reference."""
    from tensors.backend.python.kernels.manipulation.slice_scatter import (
        slice_scatter as reference,
    )

    if not _array_work_is_large_enough(
        _shape_size(output_shape), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(value, indices, output_shape=output_shape)
    slice_scatter = _backend_kernel("slice_scatter")
    result = slice_scatter(value, indices, output_shape=output_shape)
    if result is not None:
        return result
    return reference(value, indices, output_shape=output_shape)
