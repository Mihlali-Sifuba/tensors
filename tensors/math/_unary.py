"""Shared dispatch helpers for elementwise unary operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..backend import UnaryOperation, execute_unary, execute_unary_gradient
from ..tensor import Tensor

if TYPE_CHECKING:
    from .._typing import Scalar
    from ..dtype import DataType


def unary_forward(
    operation: UnaryOperation,
    value: Tensor,
    *,
    dtype: DataType,
    fallback: Callable[[Scalar], Scalar],
) -> Tensor:
    """Build a tensor from an accelerated unary kernel or scalar fallback."""
    storage = execute_unary(operation, value, dtype=dtype)
    values: Any = storage
    if values is None:
        values = [fallback(item) for item in value._data]
    return Tensor(values, dtype=dtype, shape=value.shape)


def unary_backward(
    operation: UnaryOperation,
    grad: Tensor,
    value: Tensor,
    *,
    fallback: Callable[[Scalar, Scalar], Scalar],
) -> Tensor:
    """Build a unary VJP from an accelerated kernel or scalar fallback."""
    storage = execute_unary_gradient(operation, grad, value)
    values: Any = storage
    if values is None:
        values = [
            fallback(upstream, item)
            for upstream, item in zip(grad._data, value._data)
        ]
    return Tensor(values, dtype=grad.dtype, shape=value.shape)
