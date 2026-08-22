"""Elementwise inverse hyperbolic sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor


class ArcSinh:
    """Elementwise inverse hyperbolic sine over the real numbers."""

    @staticmethod
    def forward(value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor(
            [math.asinh(float(item)) for item in value._data],
            dtype=dtype,
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
            item = float(item)
            magnitude = abs(item)
            if math.isinf(magnitude):
                derivative = 0.0
            elif magnitude <= 1.0:
                derivative = 1.0 / math.sqrt(1.0 + item * item)
            else:
                reciprocal = 1.0 / magnitude
                derivative = reciprocal / math.sqrt(1.0 + reciprocal * reciprocal)
            gradients.append(upstream * derivative)
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
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
        return Variable._from_operation(
            ArcSinh.forward(value.data),
            "arcsinh",
            ArcSinh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ArcSinh.forward(value)


__all__ = ["ArcSinh", "arcsinh"]
