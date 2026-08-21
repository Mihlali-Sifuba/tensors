"""Elementwise rectified linear unit and its differentiation rule."""

import math
from typing import Any, List

from ..tensor import Tensor


class ReLU:
    """Elementwise rectified linear unit with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        values = [
            math.nan if isinstance(value, float) and math.isnan(value)
            else value if value > 0 else 0
            for value in a._data
        ]
        return Tensor(values, dtype=a.dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [
            math.nan if isinstance(x, float) and math.isnan(x)
            else g if x > 0 else 0
            for g, x in zip(grad._data, a._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build an almost-everywhere differentiable VJP for ReLU."""
        from ..variable import Variable
        value = inputs[0]
        mask = Tensor(
            [
                math.nan if isinstance(item, float) and math.isnan(item)
                else 1.0 if item > 0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [grad * Variable(mask, requires_grad=False)]


def relu(value: Any) -> Any:
    """Return the elementwise rectified linear unit as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ReLU.forward(value.data),
            "relu",
            ReLU,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ReLU.forward(value)


__all__ = ["ReLU", "relu"]
