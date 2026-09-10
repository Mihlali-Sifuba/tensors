"""Differentiable tensor transpose."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from .dot import _transpose_impl

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class Transpose(Operation):
    """Transpose final matrix axes with a reverse-mode gradient rule."""

    __slots__ = ("axes",)
    name = "transpose"

    def __init__(self, *, axes: tuple[int, ...] | None = None) -> None:
        object.__setattr__(self, "axes", axes)

    def forward(self, value: Tensor) -> Tensor:
        """Permute the tensor axes described by this invocation."""
        return _transpose_impl(value, self.axes)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Transpose the upstream gradient back to the input layout."""
        axes = self.axes
        if axes is None:
            return [_transpose_impl(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [_transpose_impl(grad, inverse)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for transpose."""
        axes = self.axes
        if axes is None:
            return [transpose(grad)]
        normalized = tuple(axis + grad.ndim if axis < 0 else axis for axis in axes)
        inverse = tuple(normalized.index(axis) for axis in range(grad.ndim))
        return [transpose(grad, axes=inverse)]


@overload
def transpose(
    value: VariableNode,
    axes: tuple[int, ...] | list[int] | None = None,
) -> VariableNode: ...


@overload
def transpose(
    value: TensorValue,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorValue: ...


@overload
def transpose(
    value: TensorData,
    axes: tuple[int, ...] | list[int] | None = None,
) -> Tensor: ...


def transpose(
    value: TensorLike | VariableNode,
    axes: tuple[int, ...] | list[int] | None = None,
) -> TensorResult | VariableNode:
    """Permute axes of a graph value or Tensor, or transpose the last two.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The permutation is the operation's own state, so a recorded
    transpose replays the axes it was written with.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    axes = tuple(axes) if isinstance(axes, list) else axes
    operation = Transpose(axes=axes)

    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Transpose", "transpose"]
