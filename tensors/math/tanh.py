"""Elementwise hyperbolic tangent and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor


class Tanh:
    """Elementwise hyperbolic tangent with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_math.tanh(float(value)) for value in a._data]
        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        values = [
            g * _tanh_derivative(float(value))
            for g, value in zip(grad._data, a._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for hyperbolic tangent."""
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
        positive_z = exp(-2.0 * masked_value_graph(value, positive_mask))
        negative_z = exp(2.0 * masked_value_graph(value, negative_mask))
        positive = 4.0 * positive_z / ((1.0 + positive_z) ** 2) * positive_mask
        negative = 4.0 * negative_z / ((1.0 + negative_z) ** 2) * negative_mask
        return [grad * (positive + negative)]


@overload
def tanh(value: TensorValue) -> TensorValue: ...


@overload
def tanh(value: TensorData) -> Tensor: ...


def tanh(value: TensorLike) -> TensorResult:
    """Return the elementwise hyperbolic tangent as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Tanh.forward(value.data),
            "tanh",
            Tanh,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tanh.forward(value)


__all__ = ["Tanh", "tanh"]


def _tanh_derivative(value: float) -> float:
    """Return the tanh derivative without subtracting rounded values."""
    z = _math.exp(-2.0 * abs(value))
    denominator = 1.0 + z
    return 4.0 * z / (denominator * denominator)
