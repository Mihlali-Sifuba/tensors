"""Elementwise cosine and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Cos:
    """Elementwise cosine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_math.cos(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [
            -g * _math.sin(float(value))
            for g, value in zip(grad._data, a._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for cosine."""
        from .sin import sin

        return [-(grad * sin(inputs[0]))]


def cos(value: Any) -> Any:
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
