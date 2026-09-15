"""Dedicated arithmetic dispatch; resolve the selection once per call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import get_backend
from ..loading import _load_array_backend
from ..policy import _shape_size, should_accelerate_elementwise
from ..storage import Storage

if TYPE_CHECKING:
    from ..._typing import Scalar
    from ...dtype import DataType
    from ...tensor import Tensor


def execute_add(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Attempt add; None requests the caller's Python fallback."""
    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return None
    backend = _load_array_backend(selected)
    return backend.add(left, right, dtype=dtype, output_shape=output_shape)


def execute_subtract(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Attempt subtract; None requests the caller's Python fallback."""
    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return None
    backend = _load_array_backend(selected)
    return backend.subtract(left, right, dtype=dtype, output_shape=output_shape)


def execute_multiply(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Attempt multiply; None requests the caller's Python fallback."""
    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return None
    backend = _load_array_backend(selected)
    return backend.multiply(left, right, dtype=dtype, output_shape=output_shape)


def execute_divide(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Attempt divide; None requests the caller's Python fallback."""
    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return None
    backend = _load_array_backend(selected)
    return backend.divide(left, right, dtype=dtype, output_shape=output_shape)


def execute_power(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Attempt power; None requests the caller's Python fallback."""
    selected = get_backend()
    if not should_accelerate_elementwise(selected, _shape_size(output_shape)):
        return None
    backend = _load_array_backend(selected)
    return backend.power(left, right, dtype=dtype, output_shape=output_shape)
