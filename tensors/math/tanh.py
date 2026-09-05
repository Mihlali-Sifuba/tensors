"""Elementwise hyperbolic tangent and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Tanh(Operation):
    """Elementwise hyperbolic tangent with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "tanh"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "tanh",
            a,
            dtype=dtype,
            fallback=lambda value: _math.tanh(float(value)),
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "tanh",
                grad,
                a,
                fallback=lambda upstream, value: (
                    upstream * _tanh_derivative(float(value))
                ),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
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
        operation = Tanh()
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tanh().forward(value)


__all__ = ["Tanh", "tanh"]


def _tanh_derivative(value: float) -> float:
    """Return the tanh derivative without subtracting rounded values."""
    z = _math.exp(-2.0 * abs(value))
    denominator = 1.0 + z
    return 4.0 * z / (denominator * denominator)
