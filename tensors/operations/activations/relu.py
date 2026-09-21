"""Elementwise rectified linear unit and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class ReLU(Operation):
    """Elementwise rectified linear unit with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "relu"

    def forward(self, value: Tensor) -> Tensor:
        """Rectify each element, preserving the operand's dtype and shape.

        The Tensor semantics are settled here — ReLU changes neither — and
        `execute_relu` owns where the rectification runs. See
        docs/relu-semantics.md.
        """
        dtype = value.dtype
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_relu(
                value, dtype=dtype, output_shape=output_shape
            ),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        a = inputs[0]
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_relu_gradient(grad, a),
                dtype=grad.dtype,
                shape=a.shape,
            )
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build an almost-everywhere differentiable VJP for ReLU."""
        from tensors.variable import Variable

        value = inputs[0]
        mask = Tensor(
            [
                (
                    math.nan
                    if isinstance(item, float) and math.isnan(item)
                    else 1.0 if item > 0 else 0.0
                )
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


def relu(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise rectified linear unit of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

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
