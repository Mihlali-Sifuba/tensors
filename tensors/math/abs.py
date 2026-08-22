"""Elementwise absolute value and its differentiation rule."""

from __future__ import annotations

import builtins
import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor


class Abs:
    """Elementwise absolute value with a zero subgradient at zero."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        return Tensor(
            [builtins.abs(item) for item in value._data],
            dtype=value.dtype,
            shape=value.shape,
        )

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        gradients = []
        for upstream, item in zip(grad._data, value._data):
            if isinstance(item, float) and math.isnan(item):
                gradients.append(math.nan)
            elif item > 0:
                gradients.append(upstream)
            elif item < 0:
                gradients.append(-upstream)
            else:
                gradients.append(0.0)
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP using the chosen zero subgradient."""
        from ..ops._utils import masked_value_graph, zero_like_graph

        value = inputs[0]
        if any(
            isinstance(item, float) and math.isnan(item)
            for item in value.data._data
        ):
            raise ValueError(
                "Higher-order derivatives of abs are undefined at NaN"
            )
        positive_mask = Tensor(
            [
                1.0 if item > 0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        negative_mask = Tensor(
            [1.0 if item < 0 else 0.0 for item in value.data._data],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [
            masked_value_graph(grad, positive_mask)
            - masked_value_graph(grad, negative_mask)
            + zero_like_graph(value)
        ]
@overload
def abs(value: TensorValue) -> TensorValue: ...


@overload
def abs(value: TensorData) -> Tensor: ...


def abs(value: TensorLike) -> TensorResult:
    """Return the elementwise absolute value of a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Abs.forward(value.data),
            "abs",
            Abs,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Abs.forward(value)


__all__ = ["Abs", "abs"]
