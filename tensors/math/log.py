"""Elementwise natural logarithm and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Log:
    """Elementwise natural logarithm with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = []
        for value in a._data:
            if value <= 0:
                raise ValueError("log is only defined for positive values")
            values.append(_math.log(float(value)))
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [g / x for g, x in zip(grad._data, a._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for the natural logarithm."""
        return [grad / inputs[0]]


def log(value: Any) -> Any:
    """Return the elementwise natural logarithm as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Log.forward(value.data),
            "log",
            Log,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Log.forward(value)


__all__ = ["Log", "log"]
