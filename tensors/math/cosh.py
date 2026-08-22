"""Elementwise hyperbolic cosine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class Cosh:
    """Elementwise hyperbolic cosine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        values = []
        for item in value._data:
            try:
                values.append(math.cosh(float(item)))
            except OverflowError:
                values.append(math.inf)
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        from .sinh import Sinh

        value = inputs[0]
        derivative = Sinh.forward(value)
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
        """Build a differentiable VJP for hyperbolic cosine."""
        from .sinh import sinh

        return [grad * sinh(inputs[0])]


def cosh(value: Any) -> Any:
    """Return the elementwise hyperbolic cosine."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Cosh.forward(value.data),
            "cosh",
            Cosh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Cosh.forward(value)


__all__ = ["Cosh", "cosh"]
