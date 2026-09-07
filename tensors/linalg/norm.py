"""Differentiable Euclidean norm."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ..math._reduction import (
    Axis,
    immutable_axis,
    keepdims_shape,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)

if TYPE_CHECKING:
    from ..graph.node import VariableNode


def _scaled_norm(
    value: Tensor,
    group: list[int],
) -> tuple[float, list[float], float]:
    """Return a safe scale, scaled values, and their Euclidean norm."""
    values = [float(value._data[index]) for index in group]
    if any(math.isinf(item) for item in values):
        return 1.0, [math.nan] * len(values), math.inf
    if any(math.isnan(item) for item in values):
        return 1.0, [math.nan] * len(values), math.nan

    scale = max((abs(item) for item in values), default=0.0)
    if scale == 0.0:
        return 0.0, [0.0] * len(values), 0.0
    normalized = [item / scale for item in values]
    normalized_magnitude = math.sqrt(math.fsum(item * item for item in normalized))
    return scale, normalized, normalized_magnitude


class Norm(Operation):
    """Whole-tensor Euclidean norm with reverse-mode gradient rules."""

    __slots__ = ("axis", "keepdims")
    name = "norm"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, value: Tensor) -> Tensor:
        """Return Euclidean norms over one, several, or all axes."""
        axis = self.axis
        keepdims = self.keepdims
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        accelerated = execute_reduction(
            "norm",
            value,
            axes,
            keepdims=keepdims,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)
        _, output_shape, groups = reduction_groups(value, axis, keepdims)
        results = []
        for group in groups:
            scale, _, normalized_magnitude = _scaled_norm(value, group)
            results.append(scale * normalized_magnitude)
        return Tensor(results, dtype=dtype, shape=output_shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Differentiate the Euclidean norm with respect to its input."""
        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, output_shape, groups = reduction_groups(value, axis, keepdims)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        values = [0.0] * value.size
        for output_index, group in enumerate(groups):
            _, normalized, normalized_magnitude = _scaled_norm(value, group)
            if normalized_magnitude == 0:
                continue
            upstream = grad._data[output_index]
            for input_index, normalized_value in zip(group, normalized):
                values[input_index] = (
                    upstream * (normalized_value / normalized_magnitude)
                )
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable VJP for nonzero axis-aware norms."""
        from ..math.reshape import reshape
        from ..variable import Variable

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, scale_shape, groups = reduction_groups(value.data, axis, True)
        statistics = [_scaled_norm(value.data, group) for group in groups]
        if any(item[2] == 0 for item in statistics):
            raise ValueError(
                "Higher-order derivatives of norm are undefined at zero"
            )
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
        expanded_grad = grad if keepdims else reshape(
            grad, keepdims_shape(value.shape, axis)
        )
        return [
            expanded_grad * (
                normalized / norm(normalized, axis=axis, keepdims=True)
            )
        ]


@overload
def norm(
    value: VariableNode,
    axis: Axis = None,
    keepdims: bool = False,
) -> VariableNode: ...


@overload
def norm(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def norm(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def norm(
    value: TensorLike | VariableNode,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult | VariableNode:
    """Compute Euclidean norms of a graph value or Tensor.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The reduction the operation was configured with is the one every
    application uses, so a recorded norm replays the call it was written as.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    operation = Norm(axis=immutable_axis(axis), keepdims=keepdims)

    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Norm", "norm"]
