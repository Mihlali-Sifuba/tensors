"""Population variance and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_reduction, execute_reduction_gradient
from ..dtype import float64
from ..graph.operation import Operation
from ..tensor import Tensor
from ._reduction import (
    Axis,
    immutable_axis,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)
from .std import _scaled_deviations


class Variance(Operation):
    """Axis-aware population variance with stable centering."""

    __slots__ = ("axis", "keepdims")
    name = "variance"

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
            "variance",
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
                values.append(math.nan)
                continue
            scale, _, normalized_deviation = _scaled_deviations(value, group)
            deviation = scale * normalized_deviation
            values.append(deviation * deviation)
        return Tensor(values, dtype=dtype, shape=output_shape)

    def backward(self, grad: Tensor, *inputs: Tensor) -> list[Tensor]:
        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{output_shape}"
            )

        accelerated = execute_reduction_gradient(
            "variance",
            grad,
            value,
            axes,
            keepdims=keepdims,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(accelerated, dtype=grad.dtype, shape=value.shape)]

        _, _, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        gradients = [0.0] * value.size
        for output_index, group in enumerate(groups):
            if not group:
                continue
            upstream = grad._data[output_index]
            if upstream == 0:
                continue
            scale, centered, _ = _scaled_deviations(value, group)
            if all(centered_value == 0.0 for centered_value in centered):
                continue
            factor = scale * (2.0 / len(group))
            for input_index, centered_value in zip(group, centered):
                gradients[input_index] = upstream * centered_value * factor
        return [Tensor(gradients, dtype=grad.dtype, shape=value.shape)]

    def backward_graph(self, grad, *inputs):
        """Build a differentiable population-variance VJP."""
        from ..ops._utils import zero_like_graph
        from ..variable import Variable
        from .mean import mean
        from .reshape import reshape

        value = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        _, scale_shape, groups = reduction_groups(value.data, axis, True)
        count = len(groups[0]) if groups else 0
        if count == 0:
            return [zero_like_graph(value)]

        statistics = [_scaled_deviations(value.data, group) for group in groups]
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
        expanded = grad if keepdims else reshape(
            grad,
            tuple(
                1 if index in normalize_axes(value.ndim, axis) else size
                for index, size in enumerate(value.shape)
            ),
        )
        factor = (scales / count) * 2.0
        return [expanded * (normalized - center) * factor]


@overload
def variance(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def variance(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def variance(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
    """Compute population variance over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)
    if isinstance(value, Variable):
        operation = Variance(axis=axis, keepdims=keepdims)
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Variance(axis=axis, keepdims=keepdims).forward(value)


__all__ = ["Variance", "variance"]
