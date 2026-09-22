"""Maximum-value public API."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import builtins
import math
from typing import TYPE_CHECKING, Any, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.reductions import (
    Axis,
    immutable_axis,
    keepdims_shape,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Max(Operation):
    """Maximum-value operation."""

    __slots__ = ("axis", "keepdims")
    name = "max"

    def __init__(self, *, axis: Axis = None, keepdims: bool = False) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, value: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and (not keepdims):
            output_shape = (1,)
        accelerated = backend_dispatch.execute_reduce_max(
            value, axes, keepdims=keepdims, dtype=value.dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=value.dtype, shape=output_shape
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Distribute each gradient equally among tied maximum values."""
        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and (not keepdims):
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = backend_dispatch.execute_reduce_max_gradient(
            grad, value, axes, keepdims=keepdims
        )
        return [
            Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)
        ]


@overload
def max(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def max(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def max(value: TensorData, axis: Axis = None, keepdims: bool = False) -> Tensor: ...


def max(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Compute maxima over one, several, or all axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)
    operation = Max(axis=axis, keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Max", "max"]
