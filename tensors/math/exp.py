"""Elementwise exponential and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Exp:
    """Elementwise exponential with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("exp", a, dtype=dtype, fallback=_exp)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "exp",
                grad,
                a,
                fallback=lambda upstream, value: upstream * _exp(value),
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for exponentiation."""
        return [grad * exp(inputs[0])]


@overload
def exp(value: TensorValue) -> TensorValue: ...


@overload
def exp(value: TensorData) -> Tensor: ...


def exp(value: TensorLike) -> TensorResult:
    """Return the elementwise exponential as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Exp.forward(value.data),
            "exp",
            Exp,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Exp.forward(value)


__all__ = ["Exp", "exp"]


def _exp(value):
    try:
        return _math.exp(float(value))
    except OverflowError:
        return _math.inf
