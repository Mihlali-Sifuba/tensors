"""Elementwise exponential and its differentiation rule."""

from __future__ import annotations

import math as _math
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._unary import unary_backward, unary_forward

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Exp(Operation):
    """Elementwise exponential with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "exp"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("exp", a, dtype=dtype, fallback=_exp)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "exp",
                grad,
                a,
                fallback=lambda upstream, value: upstream * _exp(value),
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for exponentiation."""
        return [grad * exp(inputs[0])]


@overload
def exp(value: VariableNode) -> VariableNode: ...


@overload
def exp(value: TensorValue) -> TensorValue: ...


@overload
def exp(value: TensorData) -> Tensor: ...


def exp(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise exponential of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Exp(), (value,))
    value = as_tensor_operand(value)
    return Exp().forward(value)


__all__ = ["Exp", "exp"]


def _exp(value):
    try:
        return _math.exp(float(value))
    except OverflowError:
        return _math.inf
