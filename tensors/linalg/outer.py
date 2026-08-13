"""Differentiable outer product of two vectors."""

from typing import Any, List

from ..dtype import result_dtype
from ..tensor import Tensor


class Outer:
    """Vector outer product with reverse-mode gradient rules."""

    @staticmethod
    def forward(a: Tensor, b: Tensor) -> Tensor:
        """Return the matrix whose entries are ``a[i] * b[j]``."""
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("outer requires two 1D vectors")

        dtype = result_dtype(a.dtype, b)
        values = [left * right for left in a._data for right in b._data]
        return Tensor(values, dtype=dtype, shape=(a.size, b.size))

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate an outer product with respect to both vectors."""
        left, right = inputs
        expected_shape = (left.size, right.size)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{expected_shape}"
            )

        left_gradient = [
            sum(
                grad._data[row * right.size + column] * right._data[column]
                for column in range(right.size)
            )
            for row in range(left.size)
        ]
        right_gradient = [
            sum(
                grad._data[row * right.size + column] * left._data[row]
                for row in range(left.size)
            )
            for column in range(right.size)
        ]
        return [
            Tensor(left_gradient, dtype=grad.dtype, shape=left.shape),
            Tensor(right_gradient, dtype=grad.dtype, shape=right.shape),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for an outer product."""
        left, right = inputs
        return [grad @ right, left @ grad]


def outer(a: Any, b: Any) -> Any:
    """Return the outer product of two vectors as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(a, Variable) or isinstance(b, Variable):
        left = a if isinstance(a, Variable) else Variable(a, requires_grad=False)
        right = b if isinstance(b, Variable) else Variable(b, requires_grad=False)
        return Variable._from_operation(
            Outer.forward(left.data, right.data),
            "outer",
            Outer,
            [left, right],
        )

    left = a if isinstance(a, Tensor) else Tensor(a)
    right = b if isinstance(b, Tensor) else Tensor(b)
    return Outer.forward(left, right)


__all__ = ["Outer", "outer"]
