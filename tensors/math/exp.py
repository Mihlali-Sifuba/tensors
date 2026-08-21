"""Elementwise exponential and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Exp:
    """Elementwise exponential with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = []
        for value in a._data:
            try:
                values.append(_math.exp(float(value)))
            except OverflowError:
                values.append(_math.inf)
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        output = Exp.forward(a)
        values = [g * y for g, y in zip(grad._data, output._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for exponentiation."""
        return [grad * exp(inputs[0])]


def exp(value: Any) -> Any:
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
