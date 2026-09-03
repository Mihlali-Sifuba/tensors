"""Elementwise inverse hyperbolic cosine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..graph.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class ArcCosh(Operation):
    """Elementwise inverse hyperbolic cosine on the real interval [1, infinity)."""

    __slots__ = ()
    name = "arccosh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("arccosh", value, dtype=dtype, fallback=_arccosh)

    def backward(self, grad: Tensor, *inputs: Tensor) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("arccosh", grad, value, fallback=_gradient)]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable VJP for inverse hyperbolic cosine."""
        from .sqrt import sqrt

        value = inputs[0]
        if any(item == 1.0 for item in value.data._data):
            raise ValueError("arccosh derivative is undefined at 1")
        denominator = sqrt(value - 1.0) * sqrt(value + 1.0)
        return [grad / denominator]


@overload
def arccosh(value: TensorValue) -> TensorValue: ...


@overload
def arccosh(value: TensorData) -> Tensor: ...


def arccosh(value: TensorLike) -> TensorResult:
    """Return the elementwise inverse hyperbolic cosine."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = ArcCosh()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcCosh().forward(value)


__all__ = ["ArcCosh", "arccosh"]


def _arccosh(value):
    if value < 1.0:
        raise ValueError(
            "arccosh is only defined for values greater than or equal to 1"
        )
    return math.acosh(float(value))


def _gradient(upstream, value):
    if value == 1.0:
        raise ValueError("arccosh derivative is undefined at 1")
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    else:
        derivative = 1.0 / (
            math.sqrt(value - 1.0) * math.sqrt(value + 1.0)
        )
    return upstream * derivative
