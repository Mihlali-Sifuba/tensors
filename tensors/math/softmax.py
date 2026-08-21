"""Numerically stable softmax and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor


def _normalize_axis(tensor: Tensor, axis: int) -> int:
    """Return a valid non-negative axis for ``tensor``."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("softmax axis must be an integer")
    if axis < 0:
        axis += tensor.ndim
    if not 0 <= axis < tensor.ndim:
        raise ValueError(f"Axis {axis} out of bounds for {tensor.ndim}D tensor")
    if tensor.shape[axis] == 0:
        raise ValueError("softmax is not defined along an empty axis")
    return axis


def _axis_layout(tensor: Tensor, axis: int) -> tuple[int, int, int]:
    """Return the row-major group sizes needed to traverse ``axis``."""
    before = 1
    for dimension in tensor.shape[:axis]:
        before *= dimension
    trailing = 1
    for dimension in tensor.shape[axis + 1:]:
        trailing *= dimension
    return before, tensor.shape[axis], trailing


class Softmax:
    """Normalize values into probabilities along a chosen axis."""

    @staticmethod
    def forward(a: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
        """Compute numerically stable softmax values along ``axis``."""
        if keepdims:
            raise ValueError("softmax does not support keepdims")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [0.0] * a.size

        for group in range(before):
            group_start = group * axis_size * trailing
            for offset in range(trailing):
                positions = [group_start + offset + index * trailing for index in range(axis_size)]
                if any(
                    math.isnan(float(a._data[position]))
                    for position in positions
                ):
                    for position in positions:
                        values[position] = math.nan
                    continue
                maximum = max(a._data[position] for position in positions)
                if maximum == math.inf:
                    maxima = [
                        position
                        for position in positions
                        if a._data[position] == math.inf
                    ]
                    probability = 1.0 / len(maxima)
                    for position in positions:
                        values[position] = probability if position in maxima else 0.0
                    continue
                if maximum == -math.inf:
                    raise ValueError(
                        "softmax is undefined when every value along an axis is -inf"
                    )
                exponentials = [math.exp(a._data[position] - maximum) for position in positions]
                total = sum(exponentials)
                for position, exponential in zip(positions, exponentials):
                    values[position] = exponential / total

        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Apply the softmax Jacobian-vector product along ``axis``."""
        a = inputs[0]
        axis = kwargs.get("axis", -1)
        if not isinstance(axis, int):
            raise TypeError("softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        output = Softmax.forward(a, axis=axis)
        values = [0.0] * a.size

        for group in range(before):
            group_start = group * axis_size * trailing
            for offset in range(trailing):
                positions = [group_start + offset + index * trailing for index in range(axis_size)]
                weighted_sum = sum(
                    grad._data[position] * output._data[position]
                    for position in positions
                )
                for position in positions:
                    values[position] = output._data[position] * (
                        grad._data[position] - weighted_sum
                    )

        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build the differentiable softmax Jacobian-vector product."""
        from .sum import sum

        axis = kwargs.get("axis", -1)
        output = softmax(inputs[0], axis=axis)
        projection = sum(grad * output, axis=axis, keepdims=True)
        return [output * (grad - projection)]


def softmax(value: Any, axis: int = -1) -> Any:
    """Return softmax probabilities for a Tensor or differentiable Variable."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            Softmax.forward(value.data, axis=axis),
            "softmax",
            Softmax,
            [value],
            axis=axis,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Softmax.forward(value, axis=axis)


__all__ = ["Softmax", "softmax"]
