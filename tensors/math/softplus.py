"""Elementwise softplus and its differentiation rule."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


class Softplus:
    """Elementwise softplus with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [
            _math.log1p(_math.exp(-abs(float(value)))) + max(float(value), 0.0)
            for value in a._data
        ]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [g * _sigmoid(float(x)) for g, x in zip(grad._data, a._data)]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]


def softplus(value: Any) -> Any:
    """Return the elementwise softplus as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Softplus.forward(value.data),
            "softplus",
            Softplus,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Softplus.forward(value)


__all__ = ["Softplus", "softplus"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)
