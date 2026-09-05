"""Elementwise inverse tangent and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class ArcTan(Operation):
    """Elementwise inverse tangent over the real numbers."""

    __slots__ = ()
    name = "arctan"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "arctan",
            value,
            dtype=dtype,
            fallback=lambda item: math.atan(float(item)),
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("arctan", grad, value, fallback=_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
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


@overload
def arctan(value: TensorValue) -> TensorValue: ...


@overload
def arctan(value: TensorData) -> Tensor: ...


def arctan(value: TensorLike) -> TensorResult:
    """Return the elementwise inverse tangent in radians."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = ArcTan()
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcTan().forward(value)


__all__ = ["ArcTan", "arctan"]


def _gradient(upstream, value):
    value = float(value)
    if math.isinf(value):
        derivative = 0.0
    elif abs(value) <= 1.0:
        derivative = 1.0 / (1.0 + value * value)
    else:
        reciprocal = 1.0 / value
        square = reciprocal * reciprocal
        derivative = square / (1.0 + square)
    return upstream * derivative
