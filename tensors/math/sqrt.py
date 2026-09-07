"""Elementwise square root and its differentiation rule."""

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
def sqrt(value: VariableNode) -> VariableNode: ...


@overload
def sqrt(value: TensorValue) -> TensorValue: ...


@overload
def sqrt(value: TensorData) -> Tensor: ...


def sqrt(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise square root of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Sqrt(), (value,))
    value = as_tensor_operand(value)
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
