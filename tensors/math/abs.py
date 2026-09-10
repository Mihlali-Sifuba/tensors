"""Elementwise absolute value and its differentiation rule."""

from __future__ import annotations

import builtins
import math
from typing import TYPE_CHECKING, Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._unary import unary_backward, unary_forward

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Abs(Operation):
    """Elementwise absolute value with a zero subgradient at zero."""

    __slots__ = ()
    name = "abs"

    def forward(self, value: Tensor) -> Tensor:
        return unary_forward(
            "abs",
            value,
            dtype=value.dtype,
            fallback=builtins.abs,
        )

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        return [unary_backward("abs", grad, value, fallback=_abs_gradient)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP using the chosen zero subgradient."""
        from ..ops._utils import masked_value_graph, zero_like_graph

        value = inputs[0]
        if any(
            isinstance(item, float) and math.isnan(item)
            for item in value.data._data
        ):
            raise ValueError(
                "Higher-order derivatives of abs are undefined at NaN"
            )
        positive_mask = Tensor(
            [
                1.0 if item > 0 else 0.0
                for item in value.data._data
            ],
            dtype=grad.dtype,
            shape=value.shape,
        )
        negative_mask = Tensor(
            [1.0 if item < 0 else 0.0 for item in value.data._data],
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [
            masked_value_graph(grad, positive_mask)
            - masked_value_graph(grad, negative_mask)
            + zero_like_graph(value)
        ]
@overload
def abs(value: VariableNode) -> VariableNode: ...


@overload
def abs(value: TensorValue) -> TensorValue: ...


@overload
def abs(value: TensorData) -> Tensor: ...


def abs(
    value: TensorLike | VariableNode,
) -> TensorResult | VariableNode:
    """Return the elementwise absolute value of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Abs(), (value,))
    value = as_tensor_operand(value)
    return Abs().forward(value)


__all__ = ["Abs", "abs"]


def _abs_gradient(upstream, value):
    if isinstance(value, float) and math.isnan(value):
        return math.nan
    if value > 0:
        return upstream
    if value < 0:
        return -upstream
    return 0.0
