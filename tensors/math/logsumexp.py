"""Numerically stable log-sum-exp reduction."""

from __future__ import annotations

import math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor
from ._reduction import Axis, keepdims_shape, reduction_groups


def _group_value(a: Tensor, indices: list[int]) -> float:
    """Return ``log(sum(exp(a[indices])))`` without avoidable overflow."""
    if not indices:
        raise ValueError("logsumexp is not defined over an empty reduction")

    if any(math.isnan(float(a._data[index])) for index in indices):
        return math.nan
    maximum = max(float(a._data[index]) for index in indices)
    if maximum == math.inf:
        return math.inf
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.fsum(
        math.exp(float(a._data[index]) - maximum) for index in indices
    ))


class LogSumExp:
    """Reduce values with a stable logarithm of summed exponentials."""

    @staticmethod
    def forward(
        a: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
        _, output_shape, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        values = [_group_value(a, group) for group in groups]
        return Tensor(values, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        _, output_shape, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )

        values = [0.0] * a.size
        for output_index, group in enumerate(groups):
            if any(math.isnan(float(a._data[index])) for index in group):
                for input_index in group:
                    values[input_index] = math.nan
                continue
            maximum = max(float(a._data[index]) for index in group)
            if maximum == math.inf:
                maxima = [index for index in group if a._data[index] == math.inf]
                weight = 1.0 / len(maxima)
                for input_index in maxima:
                    values[input_index] = grad._data[output_index] * weight
                continue
            if maximum == -math.inf:
                raise ValueError(
                    "logsumexp gradient is undefined when every reduced value is -inf"
                )

            exponentials = [
                math.exp(float(a._data[index]) - maximum) for index in group
            ]
            total = math.fsum(exponentials)
            for input_index, exponential in zip(group, exponentials):
                values[input_index] = (
                    grad._data[output_index] * exponential / total
                )

        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable log-sum-exp VJP."""
        from .exp import exp
        from .reshape import reshape
        from ..variable import Variable

        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        value = inputs[0]

        # ``inf - inf`` becomes NaN in the smooth expression below.  The
        # first-order implementation already defines a stable subgradient at
        # positive infinity, so preserve that convention here as a constant
        # subgradient rather than manufacturing NaNs in a gradient graph.
        _, _, groups = reduction_groups(
            value.data,
            axis,
            keepdims=False,
            scalar_as_vector=True,
        )
        if any(
            group
            and not any(
                math.isnan(float(value.data._data[index])) for index in group
            )
            and max(float(value.data._data[index]) for index in group) == math.inf
            for group in groups
        ):
            numerical_gradient = LogSumExp.backward(
                grad.data,
                value.data,
                axis=axis,
                keepdims=keepdims,
            )[0]
            return [Variable(numerical_gradient, requires_grad=False)]

        normalizer = logsumexp(value, axis=axis, keepdims=True)
        expanded_grad = (
            grad
            if keepdims
            else reshape(grad, keepdims_shape(value.shape, axis))
        )
        return [expanded_grad * exp(value - normalizer)]


def logsumexp(
    value: Any,
    axis: Axis = None,
    keepdims: bool = False,
) -> Any:
    """Compute ``log(sum(exp(value)))`` stably over selected axes."""
    from ..variable import Variable

    if isinstance(axis, list):
        axis = tuple(axis)

    if isinstance(value, Variable):
        return Variable._from_operation(
            LogSumExp.forward(value.data, axis=axis, keepdims=keepdims),
            "logsumexp",
            LogSumExp,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return LogSumExp.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["LogSumExp", "logsumexp"]
