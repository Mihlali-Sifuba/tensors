"""Differentiable stack operation."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, List, Optional, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import execute_stack
from ..dtype import result_dtype
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand

if TYPE_CHECKING:
    from ..graph.node import VariableNode
    from ..variable import Variable


def _operands(tensors: Sequence[Any]) -> tuple[Any, ...]:
    """Read a lone list argument as the operands it holds.

    A caller may name one tensor per argument or pass a single list of
    them; both the public function and :meth:`Stack.forward` read that
    shape the same way.
    """
    if len(tensors) == 1 and isinstance(tensors[0], list):
        return tuple(tensors[0])
    return tuple(tensors)


class Stack(Operation):
    """Stack tensors along a new axis and split gradients back to inputs."""

    __slots__ = ("axis",)
    name = "stack"

    def __init__(
        self,
        *,
        axis: int = 0,
    ) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, *tensors: Tensor | list[Tensor]) -> Tensor:
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        converted: tuple[Tensor, ...] = _operands(tensors)
        if not converted:
            raise ValueError("stack requires at least one tensor")

        elem_shape = converted[0].shape
        n = len(converted)
        for t in converted[1:]:
            if t.shape != elem_shape:
                raise ValueError(
                    f"All tensors must have the same shape; got {elem_shape} and {t.shape}"
                )

        if axis < 0:
            axis += len(elem_shape) + 1
        if not (0 <= axis <= len(elem_shape)):
            raise ValueError(
                f"Axis {axis} out of bounds for {len(elem_shape)}D tensor stack"
            )

        out_shape = list(elem_shape)
        out_shape.insert(axis, n)

        before = 1
        for d in elem_shape[:axis]:
            before *= d

        axis_stride = 1
        for d in elem_shape[axis:]:
            axis_stride *= d

        dtype = converted[0].dtype
        for tensor in converted[1:]:
            dtype = result_dtype(dtype, tensor)
        accelerated = execute_stack(
            converted,
            axis,
            dtype=dtype,
            output_shape=tuple(out_shape),
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(
                accelerated,
                dtype=dtype,
                shape=tuple(out_shape),
            )
        result = array(dtype.typecode, [])
        for g in range(before):
            base = g * axis_stride
            for k in range(n):
                for t in range(axis_stride):
                    result.append(converted[k]._data[base + t])

        return Tensor(result, dtype=dtype, shape=tuple(out_shape))

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        """Select the requested slices along the inserted axis."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        gradients: List[Optional[Tensor]] = []
        for index, wanted in enumerate(needs_input_grad):
            if not wanted:
                gradients.append(None)
                continue
            key = [slice(None)] * grad.ndim
            key[axis] = index
            gradients.append(grad[tuple(key)])
        return gradients

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build differentiable selections for a stack VJP."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("stack axis must be an integer")
        if axis < 0:
            axis += grad.ndim
        gradients = []
        for index, wanted in enumerate(needs_input_grad):
            if not wanted:
                gradients.append(None)
                continue
            key = [slice(None)] * grad.ndim
            key[axis] = index
            gradients.append(grad[tuple(key)])
        return gradients


@overload
def stack(
    tensors: Sequence[VariableNode],
    axis: int = 0,
) -> VariableNode: ...


@overload
def stack(tensors: Sequence[Variable], axis: int = 0) -> Variable: ...


@overload
def stack(tensors: Sequence[TensorData], axis: int = 0) -> Tensor: ...


@overload
def stack(tensors: Sequence[TensorLike], axis: int = 0) -> TensorResult: ...


@overload
def stack(
    tensors: Sequence[TensorLike | VariableNode],
    axis: int = 0,
) -> TensorResult | VariableNode: ...


def stack(
    tensors: Sequence[TensorLike | VariableNode],
    axis: int = 0,
) -> TensorResult | VariableNode:
    """Stack a sequence of graph values, Tensors or Variables on a new axis.

    The operands are a sequence, so the graph is asked about each element
    rather than about the sequence itself. One graph value anywhere in it
    applies the whole expression through the graph, contributing one
    operand per element in the order given: Variables calculate the result
    now, a vertex records the operation for a program that runs later, and
    a Tensor beside either enters the graph as a non-gradient leaf.
    """
    from ..graph.expression import (
        apply_operation, as_graph_operand, is_graph_operand,
    )

    if any(is_graph_operand(value) for value in tensors):
        return apply_operation(
            Stack(axis=axis),
            tuple(as_graph_operand(value) for value in tensors),
        )
    return Stack(axis=axis).forward(
        *(as_tensor_operand(value) for value in _operands(tensors))
    )


__all__ = ["Stack", "stack"]
