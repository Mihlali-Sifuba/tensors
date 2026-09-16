"""Minimum-value public API."""

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


class Min(Operation):
    """Minimum-value operation."""

    __slots__ = ("axis", "keepdims")
    name = "min"

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
        accelerated = backend_dispatch.execute_reduce_min(
            value, axes, keepdims=keepdims, dtype=value.dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=value.dtype, shape=output_shape
        )

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Distribute each gradient equally among tied minimum values."""
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
        accelerated = backend_dispatch.execute_reduce_min_gradient(
            grad, value, axes, keepdims=keepdims
        )
        return [
            Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP where every minimum is unique."""
        from tensors.operations._gradient_shaping import (
            masked_value_graph,
            zero_like_graph,
        )
        from tensors.operations.manipulation.reshape import reshape

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, _, groups = reduction_groups(
            value.data.shape, axis, keepdims, scalar_as_vector=True
        )
        weights = [0.0] * value.size
        for group in groups:
            group_values = [value.data._data[index] for index in group]
            if any(
                (isinstance(item, float) and math.isnan(item) for item in group_values)
            ):
                raise ValueError("Higher-order derivatives of min are undefined at NaN")
            minimum = builtins.min(group_values)
            selected = [index for index in group if value.data._data[index] == minimum]
            if len(selected) != 1:
                raise ValueError(
                    "Higher-order derivatives of min are undefined at ties"
                )
            weights[selected[0]] = 1.0
        expanded = (
            grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        )
        mask = Tensor(weights, dtype=grad.dtype, shape=value.shape)
        return [masked_value_graph(expanded, mask) + zero_like_graph(value)]


@overload
def min(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def min(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def min(value: TensorData, axis: Axis = None, keepdims: bool = False) -> Tensor: ...


def min(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Compute minima over one, several, or all axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)
    operation = Min(axis=axis, keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Min", "min"]
