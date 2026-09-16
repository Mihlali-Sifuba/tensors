"""Population variance and its differentiation rule."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, Any, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
from tensors.ops.operation import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.reductions import (
    Axis,
    immutable_axis,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)
from tensors.utils.deviation import scaled_deviations

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode


class Variance(Operation):
    """Axis-aware population variance with stable centering."""

    __slots__ = ("axis", "keepdims")
    name = "variance"

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
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        accelerated = backend_dispatch.execute_reduce_variance(
            value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> list[Tensor]:
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
        accelerated = backend_dispatch.execute_reduce_variance_gradient(
            grad, value, axes, keepdims=keepdims
        )
        return [
            Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable population-variance VJP."""
        from tensors.ops._utils import zero_like_graph
        from tensors.variable import Variable
        from tensors.math.mean import mean
        from tensors.math.reshape import reshape

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, scale_shape, groups = reduction_groups(value.data.shape, axis, True)
        count = len(groups[0]) if groups else 0
        if count == 0:
            return [zero_like_graph(value)]
        statistics = [scaled_deviations(value.data._data, group) for group in groups]
        scales = Variable(
            Tensor(
                [
                    scale if math.isfinite(scale) and scale > 0.0 else 1.0
                    for scale, _, _ in statistics
                ],
                dtype=value.dtype,
                shape=scale_shape,
            ),
            requires_grad=False,
        )
        normalized = value / scales
        center = mean(normalized, axis=axis, keepdims=True)
        expanded = (
            grad
            if keepdims
            else reshape(
                grad,
                tuple(
                    (
                        1 if index in normalize_axes(value.ndim, axis) else size
                        for index, size in enumerate(value.shape)
                    )
                ),
            )
        )
        factor = scales / count * 2.0
        return [expanded * (normalized - center) * factor]


@overload
def variance(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def variance(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def variance(
    value: TensorData, axis: Axis = None, keepdims: bool = False
) -> Tensor: ...


def variance(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Compute population variance over one, several, or all axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)
    operation = Variance(axis=axis, keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Variance", "variance"]
