"""Elementwise hyperbolic cosine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Cosh:
    """Elementwise hyperbolic cosine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("cosh", value, dtype=dtype, fallback=_cosh)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            unary_backward(
                "cosh",
                grad,
                value,
                fallback=lambda upstream, item: upstream * _sinh(item),
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for hyperbolic cosine."""
        from .sinh import sinh

        return [grad * sinh(inputs[0])]


@overload
def cosh(value: TensorValue) -> TensorValue: ...


@overload
def cosh(value: TensorData) -> Tensor: ...


def cosh(value: TensorLike) -> TensorResult:
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


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf


def _sinh(value):
    try:
        return math.sinh(float(value))
    except OverflowError:
        return math.copysign(math.inf, value)
