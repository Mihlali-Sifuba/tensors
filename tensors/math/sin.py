"""Elementwise sine and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Sin:
    """Elementwise sine with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward(
            "sin",
            a,
            dtype=dtype,
            fallback=lambda value: _math.sin(float(value)),
        )

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
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

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
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
        return Variable._from_operation(
            Sin.forward(value.data),
            "sin",
            Sin,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sin.forward(value)


__all__ = ["Sin", "sin"]
