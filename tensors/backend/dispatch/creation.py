"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
    _shape_size,
)
from ..storage import Storage

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

def execute_full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create constant-filled storage with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    full = _backend_kernel("full")
    return full(shape, fill_value, dtype=dtype)

def execute_eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create identity-like matrix storage with an accelerated backend."""
    shape = (rows, columns)
    if not _array_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    eye = _backend_kernel("eye")
    return eye(rows, columns, k, dtype=dtype)

def execute_arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create arithmetic-progression storage with an accelerated backend."""
    if not _array_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    arange = _backend_kernel("arange")
    return arange(start, step, count, dtype=dtype)

def execute_linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create evenly spaced storage with an accelerated backend when safe."""
    if not _array_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    linspace = _backend_kernel("linspace")
    return linspace(start, stop, count, dtype=dtype)

def execute_one_hot_targets(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> Storage | None:
    """Expand class-index targets with the active array backend."""
    if not _array_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    one_hot_targets = _backend_kernel("one_hot_targets")
    return one_hot_targets(logits, targets, axis)
