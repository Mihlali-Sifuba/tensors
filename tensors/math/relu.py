"""Elementwise rectified linear unit and its differentiation rule."""

import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class ReLU:
    """Elementwise rectified linear unit with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        return unary_forward("relu", a, dtype=a.dtype, fallback=_relu)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        return [unary_backward("relu", grad, a, fallback=_gradient)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build an almost-everywhere differentiable VJP for ReLU."""
        from ..variable import Variable
        value = inputs[0]
        mask = Tensor(
            [
                math.nan if isinstance(item, float) and math.isnan(item)
                else 1.0 if item > 0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [grad * Variable(mask, requires_grad=False)]


@overload
def relu(value: TensorValue) -> TensorValue: ...


@overload
def relu(value: TensorData) -> Tensor: ...


def relu(value: TensorLike) -> TensorResult:
    """Return the elementwise rectified linear unit as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            ReLU.forward(value.data),
            "relu",
            ReLU,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return ReLU.forward(value)


__all__ = ["ReLU", "relu"]


def _relu(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value if value > 0 else 0


def _gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return upstream if value > 0 else 0
