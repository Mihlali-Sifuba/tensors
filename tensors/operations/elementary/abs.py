"""Elementwise absolute value and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Abs(Operation):
    """Elementwise absolute value with a zero subgradient at zero."""

    __slots__ = ()
    name = "abs"

    def forward(self, value: Tensor) -> Tensor:
        return Tensor._from_owned_storage(
            backend_dispatch.execute_abs(value, dtype=value.dtype),
            dtype=value.dtype,
            shape=value.shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
        value = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_abs_gradient(grad, value),
                dtype=grad.dtype,
                shape=value.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP using the chosen zero subgradient."""
        from tensors.operations._gradient_shaping import (
            masked_value_graph,
            zero_like_graph,
        )

        value = inputs[0]
        if any(
            (isinstance(item, float) and math.isnan(item) for item in value.data._data)
        ):
            raise ValueError("Higher-order derivatives of abs are undefined at NaN")
        positive_mask = Tensor(
            [1.0 if item > 0 else 0.0 for item in value.data._data],
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


def abs(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise absolute value of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

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
