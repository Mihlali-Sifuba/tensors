"""Elementwise rectified linear unit and its differentiation rule."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._unary import unary_backward, unary_forward

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class ReLU(Operation):
    """Elementwise rectified linear unit with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "relu"

    def forward(self, a: Tensor) -> Tensor:
        return unary_forward("relu", a, dtype=a.dtype, fallback=_relu)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        return [unary_backward("relu", grad, a, fallback=_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build an almost-everywhere differentiable VJP for ReLU."""
        from ..variable import Variable
        value = inputs[0]
        mask = Tensor(
            [
                math.nan if isinstance(item, float) and math.isnan(item)
                else 1.0 if item > 0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [grad * Variable(mask, requires_grad=False)]


@overload
def relu(value: VariableNode) -> VariableNode: ...


@overload
def relu(value: TensorValue) -> TensorValue: ...


@overload
def relu(value: TensorData) -> Tensor: ...


def relu(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise rectified linear unit of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(ReLU(), (value,))
    value = as_tensor_operand(value)
    return ReLU().forward(value)


__all__ = ["ReLU", "relu"]


def _relu(value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return value if value > 0 else 0


def _gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    return upstream if value > 0 else 0
