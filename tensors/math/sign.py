"""Elementwise sign and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..graph.operation import Operation
from ..tensor import Tensor
from ._unary import unary_backward, unary_forward


class Sign(Operation):
    """Elementwise sign with a zero derivative away from zero."""

    __slots__ = ()
    name = "sign"

    def forward(self, value: Tensor) -> Tensor:
        return unary_forward("sign", value, dtype=value.dtype, fallback=_sign)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("sign", grad, value, fallback=_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build the zero VJP on intervals where sign is constant."""
        from ..ops._utils import zero_like_graph

        value = inputs[0]
        if any(item == 0 for item in value.data._data):
            raise ValueError("sign derivative is undefined at zero")
        if any(
            isinstance(item, float) and math.isnan(item)
            for item in value.data._data
        ):
            raise ValueError("Higher-order derivatives of sign are undefined at NaN")
        return [zero_like_graph(value)]


@overload
def sign(value: TensorValue) -> TensorValue: ...


@overload
def sign(value: TensorData) -> Tensor: ...


def sign(value: TensorLike) -> TensorResult:
    """Return -1, 0, or 1 according to each element's sign."""
    from ..variable import Variable

    if isinstance(value, Variable):
        operation = Sign()
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Sign().forward(value)


__all__ = ["Sign", "sign"]


def _sign(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _gradient(upstream, value):
    if value == 0:
        raise ValueError("sign derivative is undefined at zero")
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return 0.0
