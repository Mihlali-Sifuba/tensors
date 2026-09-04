"""Elementwise tangent and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Tan(Operation):
    """Elementwise tangent with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "tan"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "tan",
            a,
            dtype=dtype,
            fallback=lambda value: _math.tan(float(value)),
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [unary_backward("tan", grad, a, fallback=_tan_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for tangent."""
        from .cos import cos

        return [grad / (cos(inputs[0]) ** 2.0)]


@overload
def tan(value: TensorValue) -> TensorValue: ...


@overload
def tan(value: TensorData) -> Tensor: ...


def tan(value: TensorLike) -> TensorResult:
    """Return the elementwise tangent as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = Tan()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Tan().forward(value)


__all__ = ["Tan", "tan"]


def _tan_gradient(upstream, value):
    cosine = _math.cos(float(value))
    return upstream / (cosine * cosine)
