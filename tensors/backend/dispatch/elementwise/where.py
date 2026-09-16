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
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run an elementwise broadcasting selection."""
    from tensors.backend.python.kernels.elementwise.where import where as reference

    if not _array_work_is_large_enough(
        _shape_size(output_shape), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(condition, left, right, dtype=dtype, output_shape=output_shape)
    where = _backend_kernel("where")
    result = where(condition, left, right, dtype=dtype, output_shape=output_shape)
    if result is not None:
        return result
    return reference(condition, left, right, dtype=dtype, output_shape=output_shape)
