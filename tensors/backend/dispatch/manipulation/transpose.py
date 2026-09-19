"""Dispatch for shape, layout, indexing, and representation changes."""

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


def execute_transpose(
    value: Tensor, permutation: tuple[int, ...], *, output_shape: tuple[int, ...]
) -> Storage:
    """Run a storage permutation with an accelerated backend."""
    from tensors.backend.python.kernels.manipulation.transpose import (
        transpose as reference,
    )

    if not _array_work_is_large_enough(value.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(value, permutation, output_shape=output_shape)
    transpose = _backend_kernel("transpose")
    result = transpose(value, permutation, output_shape=output_shape)
    if result is not None:
        return result
    return reference(value, permutation, output_shape=output_shape)
