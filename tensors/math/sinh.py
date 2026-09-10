"""Elementwise hyperbolic sine and its differentiation rule."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._unary import unary_backward, unary_forward

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Sinh(Operation):
    """Elementwise hyperbolic sine with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sinh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return unary_forward("sinh", value, dtype=dtype, fallback=_sinh)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            unary_backward(
                "sinh",
                grad,
                value,
                fallback=lambda upstream, item: upstream * _cosh(item),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for hyperbolic sine."""
        from .cosh import cosh

        return [grad * cosh(inputs[0])]


@overload
def sinh(value: VariableNode) -> VariableNode: ...


@overload
def sinh(value: TensorValue) -> TensorValue: ...


@overload
def sinh(value: TensorData) -> Tensor: ...


def sinh(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise hyperbolic sine.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Sinh(), (value,))
    value = as_tensor_operand(value)
    return Sinh().forward(value)


__all__ = ["Sinh", "sinh"]


def _sinh(value):
    try:
        return math.sinh(float(value))
    except OverflowError:
        return math.copysign(math.inf, value)


def _cosh(value):
    try:
        return math.cosh(float(value))
    except OverflowError:
        return math.inf
