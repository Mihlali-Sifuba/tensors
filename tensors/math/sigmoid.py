"""Elementwise sigmoid and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Sigmoid:
    """Elementwise sigmoid with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_sigmoid(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        output = Sigmoid.forward(a)
        values = [g * y * (1.0 - y) for g, y in zip(grad._data, output._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]


def sigmoid(value: Any) -> Any:
    """Return the elementwise sigmoid as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sigmoid.forward(value.data),
            "sigmoid",
            Sigmoid,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sigmoid.forward(value)


__all__ = ["Sigmoid", "sigmoid"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)
