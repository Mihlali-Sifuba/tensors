"""Elementwise tangent and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Tan:
    """Elementwise tangent with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_math.tan(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = []
        for gradient, value in zip(grad._data, a._data):
            cosine = _math.cos(float(value))
            values.append(gradient / (cosine * cosine))
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for tangent."""
        from .cos import cos

        return [grad / (cos(inputs[0]) ** 2.0)]


def tan(value: Any) -> Any:
    """Return the elementwise tangent as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Tan.forward(value.data),
            "tan",
            Tan,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tan.forward(value)


__all__ = ["Tan", "tan"]
