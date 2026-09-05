"""Elementwise inverse sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class ArcSin(Operation):
    """Elementwise inverse sine on the closed real interval [-1, 1]."""

    __slots__ = ()
    name = "arcsin"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("arcsin", value, dtype=dtype, fallback=_arcsin)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            unary_backward(
                "arcsin",
                grad,
                value,
                fallback=_arcsin_gradient,
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for inverse sine."""
        from .sqrt import sqrt

        value = inputs[0]
        if any(item == -1.0 or item == 1.0 for item in value.data._data):
            raise ValueError("arcsin derivative is undefined at -1 and 1")
        return [grad / sqrt(1.0 - value ** 2.0)]


@overload
def arcsin(value: TensorValue) -> TensorValue: ...


@overload
def arcsin(value: TensorData) -> Tensor: ...


def arcsin(value: TensorLike) -> TensorResult:
    """Return the elementwise inverse sine in radians."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = ArcSin()
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcSin().forward(value)


__all__ = ["ArcSin", "arcsin"]


def _arcsin(value):
    if value < -1.0 or value > 1.0:
        raise ValueError("arcsin is only defined for values between -1 and 1")
    return math.asin(float(value))


def _arcsin_gradient(upstream, value):
    if value == -1.0 or value == 1.0:
        raise ValueError("arcsin derivative is undefined at -1 and 1")
    return upstream / math.sqrt(1.0 - float(value) ** 2.0)
