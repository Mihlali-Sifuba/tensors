"""Elementwise inverse tangent and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class ArcTan:
    """Elementwise inverse tangent over the real numbers."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor(
            [math.atan(float(item)) for item in value._data],
            dtype=dtype,
            shape=value.shape,
        )

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        values = []
        for upstream, item in zip(grad._data, value._data):
            item = float(item)
            if math.isinf(item):
                derivative = 0.0
            elif abs(item) <= 1.0:
                derivative = 1.0 / (1.0 + item * item)
            else:
                reciprocal = 1.0 / item
                square = reciprocal * reciprocal
                derivative = square / (1.0 + square)
            values.append(upstream * derivative)
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for inverse tangent."""
        from .abs import abs

        value = inputs[0]
        scale = 1.0 + abs(value)
        reciprocal_scale = 1.0 / scale
        normalized_value = value / scale
        reciprocal_square = reciprocal_scale ** 2.0
        derivative = reciprocal_square / (
            reciprocal_square + normalized_value ** 2.0
        )
        return [grad * derivative]


def arctan(value: Any) -> Any:
    """Return the elementwise inverse tangent in radians."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ArcTan.forward(value.data),
            "arctan",
            ArcTan,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcTan.forward(value)


__all__ = ["ArcTan", "arctan"]
