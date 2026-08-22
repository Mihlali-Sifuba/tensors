"""Elementwise sigmoid and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor


class Sigmoid:
    """Elementwise sigmoid with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_sigmoid(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [
            g * _sigmoid_derivative(float(value))
            for g, value in zip(grad._data, a._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for sigmoid."""
        from ..ops._utils import masked_value_graph
        from .exp import exp

        value = inputs[0]
        positive_mask = Tensor(
            [1.0 if item >= 0.0 else 0.0 for item in value.data._data],
            dtype=value.dtype,
            shape=value.shape,
        )
        negative_mask = Tensor(
            [1.0 - item for item in positive_mask._data],
            dtype=value.dtype,
            shape=value.shape,
        )
        positive_z = exp(-masked_value_graph(value, positive_mask))
        negative_z = exp(masked_value_graph(value, negative_mask))
        positive = positive_z / ((1.0 + positive_z) ** 2) * positive_mask
        negative = negative_z / ((1.0 + negative_z) ** 2) * negative_mask
        return [grad * (positive + negative)]


@overload
def sigmoid(value: TensorValue) -> TensorValue: ...


@overload
def sigmoid(value: TensorData) -> Tensor: ...


def sigmoid(value: TensorLike) -> TensorResult:
    """Return the elementwise sigmoid as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Sigmoid.forward(value.data),
            "sigmoid",
            Sigmoid,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sigmoid.forward(value)


__all__ = ["Sigmoid", "sigmoid"]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = _math.exp(-value)
        return 1.0 / (1.0 + z)
    z = _math.exp(value)
    return z / (1.0 + z)


def _sigmoid_derivative(value: float) -> float:
    """Return the sigmoid derivative without subtracting from rounded one."""
    z = _math.exp(-value) if value >= 0.0 else _math.exp(value)
    denominator = 1.0 + z
    return z / (denominator * denominator)
