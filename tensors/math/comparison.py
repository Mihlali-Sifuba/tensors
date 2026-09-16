"""Broadcasting elementwise comparisons."""

from __future__ import annotations

from typing import Any

from tensors._typing import TensorLike
from tensors.backend import (
    execute_equal,
    execute_greater,
    execute_greater_equal,
    execute_less,
    execute_less_equal,
    execute_not_equal,
)
from tensors.backend.storage import Storage
from tensors.dtype import result_dtype, uint8
from tensors.shape import Shape
from tensors.tensor import Tensor


def _tensor(value: Any, *, reference_dtype: Any = None) -> Tensor:
    from tensors.variable import Variable

    if isinstance(value, Variable):
        return value.data
    if isinstance(value, Tensor):
        return value
    dtype = (
        result_dtype(reference_dtype, value)
        if reference_dtype is not None and isinstance(value, (int, float))
        else None
    )
    return Tensor(value, dtype=dtype)


def _operands(left: Any, right: Any) -> tuple[Tensor, Tensor, Shape]:
    """Return both comparison operands as tensors with their result shape.

    A scalar takes its dtype from the tensor it is compared against, in
    whichever position it appears.
    """
    left_tensor = _tensor(left)
    right_tensor = _tensor(right, reference_dtype=left_tensor.dtype)
    if isinstance(left, (int, float)):
        left_tensor = _tensor(left, reference_dtype=right_tensor.dtype)
    return (
        left_tensor,
        right_tensor,
        left_tensor.shape.broadcast_with(right_tensor.shape),
    )


def _mask(storage: Storage, output_shape: Shape) -> Tensor:
    return Tensor._from_owned_storage(storage, dtype=uint8, shape=output_shape)


def equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return an elementwise equality mask with dtype ``uint8``."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(execute_equal(left_tensor, right_tensor, output_shape=shape), shape)


def not_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return an elementwise inequality mask with dtype ``uint8``."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(
        execute_not_equal(left_tensor, right_tensor, output_shape=shape), shape
    )


def less(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left < right`` mask."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(execute_less(left_tensor, right_tensor, output_shape=shape), shape)


def less_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left <= right`` mask."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(
        execute_less_equal(left_tensor, right_tensor, output_shape=shape),
        shape,
    )


def greater(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left > right`` mask."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(execute_greater(left_tensor, right_tensor, output_shape=shape), shape)


def greater_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left >= right`` mask."""
    left_tensor, right_tensor, shape = _operands(left, right)
    return _mask(
        execute_greater_equal(left_tensor, right_tensor, output_shape=shape),
        shape,
    )


__all__ = ["equal", "greater", "greater_equal", "less", "less_equal", "not_equal"]
