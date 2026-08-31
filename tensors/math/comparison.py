"""Broadcasting elementwise comparisons."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._typing import TensorLike
from ..backend import ComparisonOperation, execute_comparison
from ..dtype import result_dtype, uint8
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_tensors


def _tensor(value: Any, *, reference_dtype=None) -> Tensor:
    from ..variable import Variable

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


def _compare(
    left: Any,
    right: Any,
    operation: ComparisonOperation,
    predicate: Callable[[int | float, int | float], bool],
) -> Tensor:
    left_tensor = _tensor(left)
    right_tensor = _tensor(right, reference_dtype=left_tensor.dtype)
    if isinstance(left, (int, float)):
        left_tensor = _tensor(left, reference_dtype=right_tensor.dtype)
    output_shape = left_tensor.shape.broadcast_with(right_tensor.shape)
    accelerated = execute_comparison(
        operation,
        left_tensor,
        right_tensor,
        output_shape=output_shape,
    )
    if accelerated is not None:
        return Tensor(accelerated, dtype=uint8, shape=output_shape)
    expanded_left, expanded_right = broadcast_tensors(left_tensor, right_tensor)
    return Tensor(
        [
            1 if predicate(left_value, right_value) else 0
            for left_value, right_value in zip(
                expanded_left._data,
                expanded_right._data,
            )
        ],
        dtype=uint8,
        shape=expanded_left.shape,
    )


def equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return an elementwise equality mask with dtype ``uint8``."""
    return _compare(left, right, "equal", lambda a, b: a == b)


def not_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return an elementwise inequality mask with dtype ``uint8``."""
    return _compare(left, right, "not_equal", lambda a, b: a != b)


def less(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left < right`` mask."""
    return _compare(left, right, "less", lambda a, b: a < b)


def less_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left <= right`` mask."""
    return _compare(left, right, "less_equal", lambda a, b: a <= b)


def greater(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left > right`` mask."""
    return _compare(left, right, "greater", lambda a, b: a > b)


def greater_equal(left: TensorLike, right: TensorLike) -> Tensor:
    """Return the elementwise ``left >= right`` mask."""
    return _compare(left, right, "greater_equal", lambda a, b: a >= b)


__all__ = [
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
]
