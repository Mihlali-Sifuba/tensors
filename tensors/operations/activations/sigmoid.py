"""Elementwise sigmoid and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import TYPE_CHECKING, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Sigmoid(Operation):
    """Elementwise sigmoid with a reverse-mode gradient rule."""

    __slots__ = ()
    name = "sigmoid"

    def forward(self, a: Tensor) -> Tensor:
        """Apply the logistic function, promoting an integer operand.

        The Tensor semantics are settled here — the shape is the operand's
        and an integer operand promotes to ``float64``, because the result
        has no integral values to return — and `execute_sigmoid` owns where
        the evaluation runs.
        """
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        output_shape = a.shape
        return Tensor._from_owned_storage(
            backend_dispatch.execute_sigmoid(a, dtype=dtype, output_shape=output_shape),
            dtype=dtype,
            shape=output_shape,
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Scale the upstream gradient by the sigmoid's derivative.

        Shape and dtype agreement are Tensor semantics and are settled here;
        `execute_sigmoid_gradient` owns where the evaluation runs. Only a
        floating operand reaches this method — an integer Tensor cannot
        require a gradient — so the promotion in ``forward`` has nothing to
        undo and the gradient carries the operand's own dtype.
        """
        a = inputs[0]
        if grad.shape != a.shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match value shape {a.shape}"
            )
        if grad.dtype is not a.dtype:
            raise ValueError(
                f"Gradient dtype {grad.dtype.name} does not match value dtype "
                f"{a.dtype.name}"
            )
        dtype = a.dtype
        output_shape = a.shape
        return [
            Tensor._from_owned_storage(
                backend_dispatch.execute_sigmoid_gradient(
                    grad, a, dtype=dtype, output_shape=output_shape
                ),
                dtype=dtype,
                shape=output_shape,
            )
        ]


@overload
def sigmoid(value: VariableNode) -> VariableNode: ...


@overload
def sigmoid(value: TensorValue) -> TensorValue: ...


@overload
def sigmoid(value: TensorData) -> Tensor: ...


def sigmoid(value: TensorLike | VariableNode) -> TensorResult | VariableNode:
    """Return the elementwise sigmoid of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    if is_graph_operand(value):
        return apply_operation(Sigmoid(), (value,))
    value = as_tensor_operand(value)
    return Sigmoid().forward(value)


__all__ = ["Sigmoid", "sigmoid"]
