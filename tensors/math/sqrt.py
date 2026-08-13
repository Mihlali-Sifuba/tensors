"""Elementwise square root and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Sqrt:
    """Elementwise square root with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = []
        for value in a._data:
            if value < 0:
                raise ValueError("sqrt is only defined for non-negative values")
            values.append(_math.sqrt(float(value)))
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        if any(value == 0 for value in a._data):
            raise ValueError("sqrt derivative is undefined at zero")
        output = Sqrt.forward(a)
        values = [g / (2.0 * y) for g, y in zip(grad._data, output._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for square root."""
        if any(value == 0 for value in inputs[0].data._data):
            raise ValueError("sqrt derivative is undefined at zero")
        return [grad / (2.0 * sqrt(inputs[0]))]


def sqrt(value: Any) -> Any:
    """Return the elementwise square root as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sqrt.forward(value.data),
            "sqrt",
            Sqrt,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sqrt.forward(value)


__all__ = ["Sqrt", "sqrt"]
