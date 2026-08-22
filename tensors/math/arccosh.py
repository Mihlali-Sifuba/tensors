"""Elementwise inverse hyperbolic cosine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class ArcCosh:
    """Elementwise inverse hyperbolic cosine on the real interval [1, infinity)."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        values = []
        for item in value._data:
            if item < 1.0:
                raise ValueError(
                    "arccosh is only defined for values greater than or "
                    "equal to 1"
                )
            values.append(math.acosh(float(item)))
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        if any(item == 1.0 for item in value._data):
            raise ValueError("arccosh derivative is undefined at 1")
        gradients = []
        for upstream, item in zip(grad._data, value._data):
            item = float(item)
            if math.isinf(item):
                derivative = 0.0
            else:
                derivative = 1.0 / (
                    math.sqrt(item - 1.0) * math.sqrt(item + 1.0)
                )
            gradients.append(upstream * derivative)
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for inverse hyperbolic cosine."""
        from .sqrt import sqrt

        value = inputs[0]
        if any(item == 1.0 for item in value.data._data):
            raise ValueError("arccosh derivative is undefined at 1")
        denominator = sqrt(value - 1.0) * sqrt(value + 1.0)
        return [grad / denominator]


def arccosh(value: Any) -> Any:
    """Return the elementwise inverse hyperbolic cosine."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ArcCosh.forward(value.data),
            "arccosh",
            ArcCosh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcCosh.forward(value)


__all__ = ["ArcCosh", "arccosh"]
