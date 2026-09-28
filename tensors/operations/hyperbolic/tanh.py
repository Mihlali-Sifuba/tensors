"""Elementwise hyperbolic tangent and its differentiation rule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, overload

from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.backend import dispatch as backend_dispatch
from tensors.dtype import float64
from tensors.graph.expression import as_tensor_operand
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Tanh(Operation):
    """Elementwise hyperbolic tangent with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "tanh"

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        output_shape = value.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_tanh(
                value, dtype=dtype, output_shape=output_shape
            ),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Optional[Tensor]]:
        """Apply the first-order VJP ``G * (1 - tanh(x)**2)``."""
        if not needs_input_grad[0]:
            return [None]
        value = inputs[0]
        if grad.shape != value.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match value shape {value.shape}"
            )
        if grad.dtype is not value.dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match value dtype "
                f"{value.dtype.name}"
            )
        dtype = value.dtype
        output_shape = value.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_tanh_gradient(
                    grad, value, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


@overload
def tanh(value: VariableNode) -> VariableNode: ...


@overload
def tanh(value: TensorValue) -> TensorValue: ...


@overload
def tanh(value: TensorData) -> Tensor: ...


def tanh(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise hyperbolic tangent of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Tanh(), (value,))
    value = as_tensor_operand(value)
    return Tanh().forward(value)


__all__ = ["Tanh", "tanh"]
