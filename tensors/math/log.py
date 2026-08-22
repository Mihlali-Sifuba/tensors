"""Elementwise natural logarithm and its differentiation rule."""

import math as _math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Log:
    """Elementwise natural logarithm with a reverse-mode gradient rule."""

    @staticmethod
    def forward(a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("log", a, dtype=dtype, fallback=_log)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "log",
                grad,
                a,
                fallback=lambda upstream, value: upstream / value,
            )
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP for the natural logarithm."""
        return [grad / inputs[0]]


@overload
def log(value: TensorValue) -> TensorValue: ...


@overload
def log(value: TensorData) -> Tensor: ...


def log(value: TensorLike) -> TensorResult:
    """Return the elementwise natural logarithm as a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Log.forward(value.data),
            "log",
            Log,
            [value],
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Log.forward(value)


__all__ = ["Log", "log"]


def _log(value):
    if value <= 0:
        raise ValueError("log is only defined for positive values")
    return _math.log(float(value))
