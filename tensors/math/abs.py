"""Elementwise absolute value and its differentiation rule."""

from __future__ import annotations

import builtins
import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Abs:
    """Elementwise absolute value with a zero subgradient at zero."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        return unary_forward(
            "abs",
            value,
            dtype=value.dtype,
            fallback=builtins.abs,
        )

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("abs", grad, value, fallback=_abs_gradient)]

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


def _abs_gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return upstream
    if value < 0:
        return -upstream
    return 0.0
