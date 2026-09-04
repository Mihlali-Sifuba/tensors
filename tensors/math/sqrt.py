"""Elementwise square root and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..graph.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Sqrt(Operation):
    """Elementwise square root with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sqrt"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("sqrt", a, dtype=dtype, fallback=_sqrt)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [unary_backward("sqrt", grad, a, fallback=_sqrt_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for square root."""
        if any(value == 0 for value in inputs[0].data._data):
            raise ValueError("sqrt derivative is undefined at zero")
        return [grad / (2.0 * sqrt(inputs[0]))]


@overload
def sqrt(value: TensorValue) -> TensorValue: ...


@overload
def sqrt(value: TensorData) -> Tensor: ...


def sqrt(value: TensorLike) -> TensorResult:
    """Return the elementwise square root as a Tensor or Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = Sqrt()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sqrt().forward(value)


__all__ = ["Sqrt", "sqrt"]


def _sqrt(value):
    if value < 0:
        raise ValueError("sqrt is only defined for non-negative values")
    return _math.sqrt(float(value))


def _sqrt_gradient(upstream, value):
    if value == 0:
        raise ValueError("sqrt derivative is undefined at zero")
    return upstream / (2.0 * _sqrt(value))
