"""Elementwise sigmoid and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Sigmoid(Operation):
    """Elementwise sigmoid with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sigmoid"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("sigmoid", a, dtype=dtype, fallback=_sigmoid)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "sigmoid",
                grad,
                a,
                fallback=lambda upstream, value: (
                    upstream * _sigmoid_derivative(float(value))
                ),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
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
        operation = Sigmoid()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sigmoid().forward(value)


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
