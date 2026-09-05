"""Elementwise sine and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Sin(Operation):
    """Elementwise sine with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sin"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "sin",
            a,
            dtype=dtype,
            fallback=lambda value: _math.sin(float(value)),
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
                "sin",
                grad,
                a,
                fallback=lambda upstream, value: (
                    upstream * _math.cos(float(value))
                ),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for sine."""
        from .cos import cos

        return [grad * cos(inputs[0])]


@overload
def sin(value: TensorValue) -> TensorValue: ...


@overload
def sin(value: TensorData) -> Tensor: ...


def sin(value: TensorLike) -> TensorResult:
    """Return the elementwise sine as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = Sin()
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sin().forward(value)


__all__ = ["Sin", "sin"]
