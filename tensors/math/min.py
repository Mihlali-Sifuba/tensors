"""Minimum-value public API."""

import builtins
import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..tensor import Tensor
from ._reduction import Axis, immutable_axis, keepdims_shape, reduction_groups


class Min:
    """Minimum-value operation."""

    @staticmethod
    def forward(
        value: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        if any(not group for group in groups):
            raise ValueError("Cannot compute min of empty tensor")
        values = []
        for group in groups:
            group_values = [value._data[index] for index in group]
            if any(
                isinstance(item, float) and math.isnan(item)
                for item in group_values
            ):
                values.append(math.nan)
            else:
                values.append(builtins.min(group_values))
        return Tensor(
            values,
            dtype=value.dtype,
            shape=output_shape,
        )

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Distribute each gradient equally among tied minimum values."""
        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        result = [0.0] * value.size
        for output_index, group in enumerate(groups):
            if any(
                isinstance(value._data[index], float)
                and math.isnan(value._data[index])
                for index in group
            ):
                for input_index in group:
                    result[input_index] = math.nan
                continue
            minimum = builtins.min(value._data[index] for index in group)
            selected = [index for index in group if value._data[index] == minimum]
            share = grad._data[output_index] / len(selected)
            for input_index in selected:
                result[input_index] = share
        return [Tensor(result, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable VJP where every minimum is unique."""
        from ..ops._utils import masked_value_graph, zero_like_graph
        from .reshape import reshape

        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = kwargs.get("keepdims", False)
        _, _, groups = reduction_groups(
            value.data,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        weights = [0.0] * value.size
        for group in groups:
            group_values = [value.data._data[index] for index in group]
            if any(
                isinstance(item, float) and math.isnan(item)
                for item in group_values
            ):
                raise ValueError(
                    "Higher-order derivatives of min are undefined at NaN"
                )
            minimum = builtins.min(group_values)
            selected = [
                index for index in group
                if value.data._data[index] == minimum
            ]
            if len(selected) != 1:
                raise ValueError(
                    "Higher-order derivatives of min are undefined at ties"
                )
            weights[selected[0]] = 1.0

        expanded = grad if keepdims else reshape(
            grad,
            keepdims_shape(value.shape, axis),
        )
        mask = Tensor(weights, dtype=grad.dtype, shape=value.shape)
        return [
            masked_value_graph(expanded, mask) + zero_like_graph(value)
        ]


@overload
def min(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def min(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def min(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
    """Compute minima over one, several, or all axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

    if isinstance(value, Variable):
        return Variable._from_operation(
            Min.forward(value.data, axis=axis, keepdims=keepdims),
            "min",
            Min,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Min.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Min", "min"]
