"""Numerically stable log-softmax and its differentiation rule."""

from __future__ import annotations

from typing import Any, List

import math

from ..dtype import float64
from ..tensor import Tensor
from .softmax import Softmax, _axis_layout, _normalize_axis


class LogSoftmax:
    """Normalize logits in log space along one axis."""

    @staticmethod
    def forward(a: Tensor, axis: int = -1) -> Tensor:
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [0.0] * a.size

        for group in range(before):
            group_start = group * axis_size * trailing
            for offset in range(trailing):
                positions = [
                    group_start + offset + index * trailing
                    for index in range(axis_size)
                ]
                maximum = max(float(a._data[position]) for position in positions)
                if maximum == math.inf:
                    maxima = [
                        position
                        for position in positions
                        if a._data[position] == math.inf
                    ]
                    selected = -math.log(len(maxima))
                    for position in positions:
                        values[position] = (
                            selected if position in maxima else -math.inf
                        )
                    continue
                if maximum == -math.inf:
                    raise ValueError(
                        "log_softmax is undefined when every value along an "
                        "axis is -inf"
                    )
                total = math.fsum(
                    math.exp(float(a._data[position]) - maximum)
                    for position in positions
                )
                normalizer = maximum + math.log(total)
                for position in positions:
                    values[position] = float(a._data[position]) - normalizer

        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        probabilities = Softmax.forward(a, axis=axis)

        from .sum import Sum

        total = Sum.forward(grad, axis=axis, keepdims=True)
        return [grad - probabilities * total]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable log-softmax VJP."""
        from .softmax import softmax
        from .sum import sum

        axis = kwargs.get("axis", -1)
        probabilities = softmax(inputs[0], axis=axis)
        return [grad - probabilities * sum(grad, axis=axis, keepdims=True)]


def log_softmax(value: Any, axis: int = -1) -> Any:
    """Return stable log probabilities along ``axis``."""
    from ..variable import Variable

    if isinstance(value, Variable):
        return Variable._from_operation(
            LogSoftmax.forward(value.data, axis=axis),
            "log_softmax",
            LogSoftmax,
            [value],
            axis=axis,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return LogSoftmax.forward(value, axis=axis)


__all__ = ["LogSoftmax", "log_softmax"]
