"""Elementwise cosine and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Cos:
    """Elementwise cosine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "cos",
            a,
            dtype=dtype,
            fallback=lambda value: _math.cos(float(value)),
        )

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "cos",
                grad,
                a,
                fallback=lambda upstream, value: (
                    -upstream * _math.sin(float(value))
                ),
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for cosine."""
        from .sin import sin

        return [-(grad * sin(inputs[0]))]


@overload
def cos(value: TensorValue) -> TensorValue: ...


@overload
def cos(value: TensorData) -> Tensor: ...


def cos(value: TensorLike) -> TensorResult:
    """Return the elementwise cosine as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Cos.forward(value.data),
            "cos",
            Cos,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Cos.forward(value)


__all__ = ["Cos", "cos"]
