"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations
from collections.abc import Sequence
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


def execute_concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Join tensor storage along an existing axis with an accelerated backend."""
    from tensors.backend.python.kernels.manipulation.concat import concat as reference

    if not _array_work_is_large_enough(
        _shape_size(output_shape), _NUMPY_ELEMENTWISE_MIN_SIZE
    ):
        return reference(values, axis, dtype=dtype, output_shape=output_shape)
    concat = _backend_kernel("concat")
    result = concat(values, axis, dtype=dtype, output_shape=output_shape)
    if result is not None:
        return result
    return reference(values, axis, dtype=dtype, output_shape=output_shape)
