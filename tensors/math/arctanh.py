"""Elementwise inverse hyperbolic tangent and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class ArcTanh:
    """Elementwise inverse hyperbolic tangent on the real interval (-1, 1)."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        values = []
        for item in value._data:
            item = float(item)
            if not math.isnan(item) and not -1.0 < item < 1.0:
                raise ValueError(
                    "arctanh is only defined for values strictly between "
                    "-1 and 1"
                )
            values.append(math.atanh(item))
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        gradients = [
            upstream / (1.0 - float(item) * float(item))
            for upstream, item in zip(grad._data, value._data)
        ]
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for inverse hyperbolic tangent."""
        value = inputs[0]
        return [grad / (1.0 - value ** 2.0)]


def arctanh(value: Any) -> Any:
    """Return the elementwise inverse hyperbolic tangent."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ArcTanh.forward(value.data),
            "arctanh",
            ArcTanh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcTanh.forward(value)


__all__ = ["ArcTanh", "arctanh"]
