"""Standard-deviation public API."""

from __future__ import annotations

import math as _math
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction, execute_reduction_gradient
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._reduction import (
    Axis,
    immutable_axis,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)
from .mean import _stable_float_mean

if TYPE_CHECKING:
    from ..graph.node import VariableNode


def _scaled_deviations(
    value: Tensor,
    group: list[int],
) -> tuple[float, list[float], float]:
    """Return a safe scale, centered scaled values, and their deviation."""
    values = [float(value._data[index]) for index in group]
    if any(not _math.isfinite(item) for item in values):
        return _math.nan, [_math.nan] * len(values), _math.nan

    count = len(values)
    average = _stable_float_mean(values)
    centered = [item - average for item in values]
    if all(_math.isfinite(item) for item in centered):
        scale = max((abs(item) for item in centered), default=0.0)
        if scale == 0.0:
            return 0.0, [0.0] * count, 0.0
        normalized_centered = [item / scale for item in centered]
    else:
        scale = max((abs(item) for item in values), default=0.0)
        normalized = [item / scale for item in values]
        normalized_average = _stable_float_mean(normalized)
        normalized_centered = [item - normalized_average for item in normalized]

    variance = _math.fsum(
        item * item / count for item in normalized_centered
    )
    return scale, normalized_centered, _math.sqrt(variance)


class Std(Operation):
    """Population standard-deviation operation."""

    __slots__ = ("axis", "keepdims")
    name = "std"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, value: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        accelerated = execute_reduction(
            "std",
            value,
            axes,
            keepdims=keepdims,
            dtype=dtype,
            output_shape=output_shape,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        values = []
        for group in groups:
            if not group:
                values.append(_math.nan)
                continue
            scale, _, normalized_deviation = _scaled_deviations(value, group)
            values.append(scale * normalized_deviation)
        return Tensor(values, dtype=dtype, shape=output_shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        """Differentiate the population standard deviation by reduction group."""
        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        accelerated = execute_reduction_gradient(
            "std",
            grad,
            value,
            axes,
            keepdims=keepdims,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)]
        _, _, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        result = [0.0] * value.size
        for output_index, group in enumerate(groups):
            if not group:
                continue
            _, centered, normalized_deviation = _scaled_deviations(value, group)
            if normalized_deviation == 0:
                continue
            normalizer = len(group) * normalized_deviation
            upstream = grad._data[output_index]
            for input_index, centered_value in zip(group, centered):
                result[input_index] = upstream * (centered_value / normalizer)
        return [Tensor(result, dtype=grad.dtype, shape=value.shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable population-standard-deviation VJP."""
        from ..ops._utils import zero_like_graph
        from ..variable import Variable
        from .mean import mean
        from .reshape import reshape

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, scale_shape, groups = reduction_groups(value.data, axis, True)
        statistics = [_scaled_deviations(value.data, group) for group in groups]
        count = len(groups[0]) if groups else 0
        if count == 0:
            return [zero_like_graph(value)]
        if count == 1:
            return [value * 0.0]
        if any(
            group and normalized_deviation == 0
            for group, (_, _, normalized_deviation) in zip(groups, statistics)
        ):
            raise ValueError(
                "Higher-order derivatives of std are undefined at zero deviation"
            )
        scales = Variable(
            Tensor(
                [
                    scale if _math.isfinite(scale) and scale > 0.0 else 1.0
                    for scale, _, _ in statistics
                ],
                dtype=value.dtype,
                shape=scale_shape,
            ),
            requires_grad=False,
        )
        normalized = value / scales
        center = mean(normalized, axis=axis, keepdims=True)
        deviation = std(normalized, axis=axis, keepdims=True)
        expanded = grad if keepdims else reshape(
            grad,
            tuple(1 if index in normalize_axes(value.ndim, axis) else size
                  for index, size in enumerate(value.shape)),
        )
        return [expanded * (normalized - center) / (count * deviation)]


@overload
def std(
    value: VariableNode,
    axis: Axis = None,
    keepdims: bool = False,
) -> VariableNode: ...


@overload
def std(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def std(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def std(
    value: TensorLike | VariableNode,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult | VariableNode:
    """Compute population standard deviation over selected axes.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    axis = immutable_axis(axis)

    operation = Std(axis=axis, keepdims=keepdims)

    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["Std", "std"]
