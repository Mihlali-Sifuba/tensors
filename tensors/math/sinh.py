"""Elementwise hyperbolic sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class Sinh:
    """Elementwise hyperbolic sine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        values = []
        for item in value._data:
            try:
                values.append(math.sinh(float(item)))
            except OverflowError:
                values.append(math.copysign(math.inf, item))
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        from .cosh import Cosh

        value = inputs[0]
        derivative = Cosh.forward(value)
        return [Tensor(
            [
                upstream * weight
                for upstream, weight in zip(grad._data, derivative._data)
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for hyperbolic sine."""
        from .cosh import cosh

        return [grad * cosh(inputs[0])]


def sinh(value: Any) -> Any:
    """Return the elementwise hyperbolic sine."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sinh.forward(value.data),
            "sinh",
            Sinh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sinh.forward(value)


__all__ = ["Sinh", "sinh"]
