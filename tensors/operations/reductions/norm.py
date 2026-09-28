"""Differentiable Euclidean norm."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
import math
from typing import TYPE_CHECKING, List, overload
from tensors._typing import TensorData, TensorLike, TensorResult, TensorValue
from tensors.dtype import float64
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


class Norm(Operation):
    """Whole-tensor Euclidean norm with reverse-mode gradient rules."""

    __slots__ = ("axis", "keepdims")
    name = "norm"

    def __init__(self, *, axis: Axis = None, keepdims: bool = False) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, value: Tensor) -> Tensor:
        """Return Euclidean norms over one, several, or all axes."""
        axis = self.axis
        keepdims = self.keepdims
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        accelerated = backend_dispatch.execute_reduce_norm(
            value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Tensor]:
        """Differentiate the Euclidean norm with respect to its input."""
        value = inputs[0]
        axes = normalize_axes(value.ndim, self.axis)
        output_shape = reduction_shape(value.shape, axes, self.keepdims)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        storage = backend_dispatch.execute_reduce_norm_gradient(
            grad, value, axes, keepdims=self.keepdims
        )
        return [
            Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=value.shape)
        ]


@overload
def norm(
    value: VariableNode, axis: Axis = None, keepdims: bool = False
) -> VariableNode: ...


@overload
def norm(
    value: TensorValue, axis: Axis = None, keepdims: bool = False
) -> TensorValue: ...


@overload
def norm(value: TensorData, axis: Axis = None, keepdims: bool = False) -> Tensor: ...


def norm(
    value: TensorLike | VariableNode, axis: Axis = None, keepdims: bool = False
) -> TensorResult | VariableNode:
    """Compute Euclidean norms of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The reduction the operation was configured with is the one every
    application uses, so a recorded norm replays the call it was written as.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    operation = Norm(axis=immutable_axis(axis), keepdims=keepdims)
    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Norm", "norm"]
