"""Standard-deviation public API."""

import builtins
import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor
from ._reduction import Axis, reduction_groups


class Std:
    """Population standard-deviation operation."""

    @staticmethod
    def forward(
        value: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        values = []
        for group in groups:
            if not group:
                values.append(0.0)
                continue
            average = builtins.sum(value._data[index] for index in group) / len(group)
            variance = builtins.sum(
                (value._data[index] - average) ** 2 for index in group
            ) / len(group)
            values.append(_math.sqrt(variance))
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor(values, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate the population standard deviation by reduction group."""
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
            if not group:
                continue
            average = builtins.sum(value._data[index] for index in group) / len(group)
            variance = builtins.sum(
                (value._data[index] - average) ** 2 for index in group
            ) / len(group)
            deviation = _math.sqrt(variance)
            if deviation == 0:
                continue
            scale = grad._data[output_index] / (len(group) * deviation)
            for input_index in group:
                result[input_index] = scale * (value._data[input_index] - average)
        return [Tensor(result, dtype=grad.dtype, shape=value.shape)]


def std(value: Any, axis: Axis = None, keepdims: bool = False) -> Any:
    """Compute population standard deviation over selected axes."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Std.forward(value.data, axis=axis, keepdims=keepdims),
            "std",
            Std,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Std.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Std", "std"]
