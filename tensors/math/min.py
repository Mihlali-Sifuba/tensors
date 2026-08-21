"""Minimum-value public API."""

import builtins
import math
from typing import Any, List

from ..tensor import Tensor
from ._reduction import Axis, immutable_axis, reduction_groups


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
        keepdims = bool(kwargs.get("keepdims", False))
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


def min(value: Any, axis: Axis = None, keepdims: bool = False) -> Any:
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
