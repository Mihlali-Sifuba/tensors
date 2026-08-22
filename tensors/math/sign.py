"""Elementwise sign and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor


class Sign:
    """Elementwise sign with a zero derivative away from zero."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        values = []
        for item in value._data:
            if isinstance(item, float) and math.isnan(item):
                values.append(math.nan)
            elif item > 0:
                values.append(1)
            elif item < 0:
                values.append(-1)
            else:
                values.append(0)
        return Tensor(values, dtype=value.dtype, shape=value.shape)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        if any(item == 0 for item in value._data):
            raise ValueError("sign derivative is undefined at zero")
        gradients = [
            math.nan
            if isinstance(item, float) and math.isnan(item)
            else 0.0
            for item in value._data
        ]
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build the zero VJP on intervals where sign is constant."""
        from ..ops._utils import zero_like_graph

        value = inputs[0]
        if any(item == 0 for item in value.data._data):
            raise ValueError("sign derivative is undefined at zero")
        if any(
            isinstance(item, float) and math.isnan(item)
            for item in value.data._data
        ):
            raise ValueError("Higher-order derivatives of sign are undefined at NaN")
        return [zero_like_graph(value)]


@overload
def sign(value: TensorValue) -> TensorValue: ...


@overload
def sign(value: TensorData) -> Tensor: ...


def sign(value: TensorLike) -> TensorResult:
    """Return -1, 0, or 1 according to each element's sign."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sign.forward(value.data),
            "sign",
            Sign,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sign.forward(value)


__all__ = ["Sign", "sign"]
