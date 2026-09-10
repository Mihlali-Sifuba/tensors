"""Dispatch for shape, layout, indexing, and representation changes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
    _shape_size,
)
from ..storage import Storage

if TYPE_CHECKING:
    from ..._typing import TensorIndex
    from ...dtype import DataType
    from ...tensor import Tensor

def execute_slice(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run accelerated tensor slicing or request the Python fallback."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    slice_tensor = _backend_kernel("slice_tensor")
    return slice_tensor(value, key, output_shape=output_shape)

def execute_slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run accelerated slice scattering or request the Python fallback."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    slice_scatter = _backend_kernel("slice_scatter")
    return slice_scatter(value, indices, output_shape=output_shape)

def execute_cast(
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run accelerated dtype conversion or request the Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    cast_tensor = _backend_kernel("cast_tensor")
    return cast_tensor(value, dtype=dtype)

def execute_transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a storage permutation with an accelerated backend."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    transpose = _backend_kernel("transpose")
    return transpose(value, permutation, output_shape=output_shape)

def execute_concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Join tensor storage along an existing axis with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    concat = _backend_kernel("concat")
    return concat(values, axis, dtype=dtype, output_shape=output_shape)

def execute_stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Join tensor storage along a new axis with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    stack = _backend_kernel("stack")
    return stack(values, axis, dtype=dtype, output_shape=output_shape)
