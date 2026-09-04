"""Differentiable outer product of two vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_outer, execute_outer_gradient
from ..dtype import result_dtype
from ..math.sum import _stable_product_sum
from ..ops.operation import Operation
from ..tensor import Tensor

if TYPE_CHECKING:
    from ..variable import Variable


class Outer(Operation):
    """Vector outer product with reverse-mode gradient rules."""

    __slots__ = ()
    name = "outer"

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """Return the matrix whose entries are ``a[i] * b[j]``."""
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("outer requires two 1D vectors")

        dtype = result_dtype(a.dtype, b)
        accelerated = execute_outer(a, b, dtype=dtype)
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=dtype,
                shape=(a.size, b.size),
            )
        values = [left * right for left in a._data for right in b._data]
        return Tensor(values, dtype=dtype, shape=(a.size, b.size))

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        """Differentiate an outer product with respect to requested vectors."""
        left, right = inputs
        need_left, need_right = needs_input_grad
        expected_shape = (left.size, right.size)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{expected_shape}"
            )
        accelerated = execute_outer_gradient(
            grad,
            left,
            right,
            needs_input_grad=needs_input_grad,
        )
        if accelerated is not None:
            left_storage, right_storage = accelerated
            return [
                Tensor._from_owned_storage(left_storage, dtype=grad.dtype, shape=left.shape)
                if left_storage is not None
                else None,
                Tensor._from_owned_storage(right_storage, dtype=grad.dtype, shape=right.shape)
                if right_storage is not None
                else None,
            ]

        left_gradient = [] if not need_left else [
            _stable_product_sum(
                [
                    (
                        float(grad._data[row * right.size + column]),
                        float(right._data[column]),
                    )
                    for column in range(right.size)
                ]
            )
            for row in range(left.size)
        ]
        right_gradient = [] if not need_right else [
            _stable_product_sum(
                [
                    (
                        float(grad._data[row * right.size + column]),
                        float(left._data[row]),
                    )
                    for row in range(left.size)
                ]
            )
            for column in range(right.size)
        ]
        return [
            Tensor(left_gradient, dtype=grad.dtype, shape=left.shape)
            if need_left
            else None,
            Tensor(right_gradient, dtype=grad.dtype, shape=right.shape)
            if need_right
            else None,
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for an outer product."""
        left, right = inputs
        need_left, need_right = needs_input_grad
        return [
            grad @ right if need_left else None,
            left @ grad if need_right else None,
        ]


@overload
def outer(a: Variable, b: TensorLike) -> Variable: ...


@overload
def outer(a: TensorLike, b: Variable) -> Variable: ...


@overload
def outer(a: TensorData, b: TensorData) -> Tensor: ...


def outer(a: TensorLike, b: TensorLike) -> TensorResult:
    """Return the outer product of two vectors as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(a, Variable) or isinstance(b, Variable):
        left = a if isinstance(a, Variable) else Variable(a, requires_grad=False)
        right = b if isinstance(b, Variable) else Variable(b, requires_grad=False)
        operation = Outer()
        return Variable._from_operation(
            operation.forward(left.data, right.data),
            operation,
            (left, right),
        )

    left = a if isinstance(a, Tensor) else Tensor(a)
    right = b if isinstance(b, Tensor) else Tensor(b)
    return Outer().forward(left, right)


__all__ = ["Outer", "outer"]
