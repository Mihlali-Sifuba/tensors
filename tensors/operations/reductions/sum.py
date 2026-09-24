"""Sum and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import TYPE_CHECKING, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.reductions import (
    Axis,
    immutable_axis,
    keepdims_shape,
    normalize_axes,
    reduction_shape,
)

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Sum(Operation):
    """Sum with a reverse-mode gradient rule."""

    __slots__ = ("axis", "keepdims")
    name = "sum"

    def __init__(self, *, axis: Axis = None, keepdims: bool = False) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        # Reducing every axis without keeping them answers as a one-element
        # vector rather than as a rank-zero value.
        if axis is None and (not keepdims):
            output_shape = (1,)
        accelerated = backend_dispatch.execute_reduce_sum(
            a, axes, keepdims=keepdims, dtype=a.dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=a.dtype, shape=output_shape
        )

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Spread the gradient back over the positions the sum consumed.

        Every input value contributes to exactly one output value, so the
        derivative of a sum with respect to any input is one and the whole
        VJP is the upstream gradient broadcast back to the input's shape. A
        reduction that did not keep its axes dropped them from the gradient,
        so they are restored first; the multiplication by ones is what
        performs the broadcast, and it is exact, because ``x * 1`` is ``x``
        for every value a gradient can hold, infinities and NaN included.

        This is the only derivative the sum defines. It is written against
        operations rather than against Tensors, so the operands decide what
        the statements mean: given Tensors they calculate, and given
        Variables the same statements record a differentiable graph. That is
        what lets the reduction inside the addition VJP be differentiated
        again.

        Both statements stay on the selected backend. Reshaping gathers in
        the tensor's own storage, and multiplication is inside the arithmetic
        execution contract, so neither consults a workload policy nor answers
        from the Python reference.
        """
        from tensors.creation import ones
        from tensors.operations.manipulation.reshape import reshape

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
        expanded = (
            grad if keepdims else reshape(grad, keepdims_shape(value.shape, axis))
        )
        return [expanded * ones(value.shape, dtype=grad.dtype)]


@overload
def sum(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def sum(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def sum(value: TensorData, axis: Axis = None, keepdims: bool = False) -> Tensor: ...


def sum(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Sum over one, several, or all axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)
    operation = Sum(axis=axis, keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Sum", "sum"]
