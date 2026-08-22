"""Elementwise softplus and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Softplus:
    """Elementwise softplus with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("softplus", a, dtype=dtype, fallback=_softplus)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "softplus",
                grad,
                a,
                fallback=lambda upstream, value: (
                    upstream * _sigmoid(float(value))
                ),
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for softplus."""
        from .sigmoid import sigmoid
        return [grad * sigmoid(inputs[0])]


@overload
def softplus(value: TensorValue) -> TensorValue: ...


@overload
def softplus(value: TensorData) -> Tensor: ...


def softplus(value: TensorLike) -> TensorResult:
    """Return the elementwise softplus as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Softplus.forward(value.data),
            "softplus",
            Softplus,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Softplus.forward(value)


__all__ = ["Softplus", "softplus"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)


def _softplus(value):
    value = float(value)
    return _math.log1p(_math.exp(-abs(value))) + max(value, 0.0)
