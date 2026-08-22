"""Elementwise inverse sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import float64
from ..tensor import Tensor


class ArcSin:
    """Elementwise inverse sine on the closed real interval [-1, 1]."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        values = []
        for item in value._data:
            if item < -1.0 or item > 1.0:
                raise ValueError("arcsin is only defined for values between -1 and 1")
            values.append(math.asin(float(item)))
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        if any(item == -1.0 or item == 1.0 for item in value._data):
            raise ValueError("arcsin derivative is undefined at -1 and 1")
        values = [
            upstream / math.sqrt(1.0 - float(item) ** 2.0)
            for upstream, item in zip(grad._data, value._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for inverse sine."""
        from .sqrt import sqrt

        value = inputs[0]
        if any(item == -1.0 or item == 1.0 for item in value.data._data):
            raise ValueError("arcsin derivative is undefined at -1 and 1")
        return [grad / sqrt(1.0 - value ** 2.0)]


def arcsin(value: Any) -> Any:
    """Return the elementwise inverse sine in radians."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ArcSin.forward(value.data),
            "arcsin",
            ArcSin,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcSin.forward(value)


__all__ = ["ArcSin", "arcsin"]
