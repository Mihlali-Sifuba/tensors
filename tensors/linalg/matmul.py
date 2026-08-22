"""Matrix-multiplication public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..tensor import Tensor
from .dot import dot

if TYPE_CHECKING:
    from ..variable import Variable


@overload
def matmul(a: Variable, b: TensorLike) -> Variable: ...


@overload
def matmul(a: TensorLike, b: Variable) -> Variable: ...


@overload
def matmul(a: TensorData, b: TensorData) -> Tensor: ...


def matmul(a: TensorLike, b: TensorLike) -> TensorResult:
    """Return the general matrix product of two tensors or Variables."""
    return dot(a, b)


__all__ = ["matmul"]
