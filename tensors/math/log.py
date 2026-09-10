"""Elementwise natural logarithm and its differentiation rule."""

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


class Log(Operation):
    """Elementwise natural logarithm with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "log"

    def forward(self, a: Tensor) -> Tensor:
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        return unary_forward("log", a, dtype=dtype, fallback=_log)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            unary_backward(
                "log",
                grad,
                a,
                fallback=lambda upstream, value: upstream / value,
            )
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for the natural logarithm."""
        return [grad / inputs[0]]


@overload
def log(value: VariableNode) -> VariableNode: ...


@overload
def log(value: TensorValue) -> TensorValue: ...


@overload
def log(value: TensorData) -> Tensor: ...


def log(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise natural logarithm of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Log(), (value,))
    value = as_tensor_operand(value)
    return Log().forward(value)


__all__ = ["Log", "log"]


def _log(value):
    if value <= 0:
        raise ValueError("log is only defined for positive values")
    return _math.log(float(value))
