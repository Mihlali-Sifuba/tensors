"""Elementwise hyperbolic tangent and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Tanh:
    """Elementwise hyperbolic tangent with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_math.tanh(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        output = Tanh.forward(a)
        values = [g * (1.0 - y * y) for g, y in zip(grad._data, output._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for hyperbolic tangent."""
        output = tanh(inputs[0])
        return [grad * (1.0 - output ** 2)]


def tanh(value: Any) -> Any:
    """Return the elementwise hyperbolic tangent as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Tanh.forward(value.data),
            "tanh",
            Tanh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tanh.forward(value)


__all__ = ["Tanh", "tanh"]
