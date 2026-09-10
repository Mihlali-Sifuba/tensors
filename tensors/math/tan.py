"""Elementwise tangent and its differentiation rule."""

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
def tan(value: VariableNode) -> VariableNode: ...


@overload
def tan(value: TensorValue) -> TensorValue: ...


@overload
def tan(value: TensorData) -> Tensor: ...


def tan(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise tangent of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Tan(), (value,))
    value = as_tensor_operand(value)
    return Tan().forward(value)


__all__ = ["Tan", "tan"]


def _tan_gradient(upstream, value):
    cosine = _math.cos(float(value))
    return upstream / (cosine * cosine)
