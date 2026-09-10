"""Matrix-multiplication public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..tensor import Tensor
from .dot import dot

if TYPE_CHECKING:
    from ..graph.node import VariableNode
    from ..variable import Variable


@overload
def matmul(a: VariableNode, b: TensorLike | VariableNode) -> VariableNode: ...


@overload
def matmul(a: TensorLike, b: VariableNode) -> VariableNode: ...


@overload
def matmul(a: Variable, b: TensorLike) -> Variable: ...


@overload
def matmul(a: TensorLike, b: Variable) -> Variable: ...


@overload
def matmul(a: TensorData, b: TensorData) -> Tensor: ...


def matmul(
    a: TensorLike | VariableNode,
    b: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the general matrix product of two graph values or tensors."""
    return dot(a, b)


__all__ = ["matmul"]
