"""Elementwise inverse hyperbolic sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..graph.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class ArcSinh(Operation):
    """Elementwise inverse hyperbolic sine over the real numbers."""

    __slots__ = ()
    name = "arcsinh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "arcsinh",
            value,
            dtype=dtype,
            fallback=lambda item: math.asinh(float(item)),
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("arcsinh", grad, value, fallback=_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a stable differentiable VJP for inverse hyperbolic sine."""
        from .abs import abs
        from .sqrt import sqrt
        from .where import where

        value = inputs[0]
        scale = 1.0 + abs(value)
        reciprocal_scale = 1.0 / scale
        normalized_value = value / scale
        stable_derivative = reciprocal_scale / sqrt(
            reciprocal_scale ** 2.0 + normalized_value ** 2.0
        )
        direct_derivative = 1.0 / sqrt(1.0 + value ** 2.0)
        large_mask = Tensor(
            [
                1.0 if math.fabs(float(item)) > 1.0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [
            grad * where(
                large_mask,
                stable_derivative,
                direct_derivative,
            )
        ]


@overload
def arcsinh(value: TensorValue) -> TensorValue: ...


@overload
def arcsinh(value: TensorData) -> Tensor: ...


def arcsinh(value: TensorLike) -> TensorResult:
    """Return the elementwise inverse hyperbolic sine."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = ArcSinh()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcSinh().forward(value)


__all__ = ["ArcSinh", "arcsinh"]


def _gradient(upstream, value):
    value = float(value)
    magnitude = abs(value)
    if math.isinf(magnitude):
        derivative = 0.0
    elif magnitude <= 1.0:
        derivative = 1.0 / math.sqrt(1.0 + value * value)
    else:
        reciprocal = 1.0 / magnitude
        derivative = reciprocal / math.sqrt(1.0 + reciprocal * reciprocal)
    return upstream * derivative
