"""Elementwise cosine and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Cos(Operation):
    """Elementwise cosine with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "cos"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "cos",
            a,
            dtype=dtype,
            fallback=lambda value: _math.cos(float(value)),
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
                "cos",
                grad,
                a,
                fallback=lambda upstream, value: (
                    -upstream * _math.sin(float(value))
                ),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for cosine."""
        from .sin import sin

        return [-(grad * sin(inputs[0]))]


@overload
def cos(value: TensorValue) -> TensorValue: ...


@overload
def cos(value: TensorData) -> Tensor: ...


def cos(value: TensorLike) -> TensorResult:
    """Return the elementwise cosine as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = Cos()
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Cos().forward(value)


__all__ = ["Cos", "cos"]
