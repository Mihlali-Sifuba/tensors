"""Reshape operation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..shape import Shape
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Reshape(Operation):
    """Reshape operation."""

    __slots__ = ("shape",)
    name = "reshape"

    def __init__(self, *, shape: Tuple[int, ...]) -> None:
        object.__setattr__(self, "shape", tuple(shape))

    def forward(self, tensor: Tensor) -> Tensor:
        shape = self.shape
        current_element_count = tensor.shape.size
        requested_element_count = Shape.from_iterable(shape).size
        if current_element_count != requested_element_count:
            raise ValueError(
                f"Cannot reshape tensor of size {current_element_count} "
                f"to shape {shape}"
            )
        return Tensor(tensor._data, dtype=tensor.dtype, shape=shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Restore the input shape without changing gradient values."""
        return [Reshape(shape=inputs[0].shape).forward(grad)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable reshape VJP."""
        return [reshape(grad, inputs[0].shape)]


@overload
def reshape(
    tensor: VariableNode,
    shape: tuple[int, ...],
) -> VariableNode: ...


@overload
def reshape(tensor: TensorValue, shape: tuple[int, ...]) -> TensorValue: ...


@overload
def reshape(tensor: TensorData, shape: tuple[int, ...]) -> Tensor: ...


def reshape(
    tensor: TensorLike | VariableNode,
    shape: tuple[int, ...],
) -> TensorResult | VariableNode:
    """Reshape a graph value or Tensor without changing its values.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The target shape is the operation's own state, so a recorded
    reshape replays the shape it was written with.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    shape = tuple(shape)
    operation = Reshape(shape=shape)

    if is_graph_operand(tensor):
        return apply_operation(operation, (tensor,))
    return operation.forward(as_tensor_operand(tensor))


__all__ = ["Reshape", "reshape"]
