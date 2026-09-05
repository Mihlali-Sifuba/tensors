"""Numerically stable log-sum-exp reduction."""

from __future__ import annotations

import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_logsumexp, execute_logsumexp_gradient
from ..dtype import float64
from ..ops.operation import Operation, UNARY_DEMAND
from ..tensor import Tensor
from ._reduction import (
    Axis,
    keepdims_shape,
    normalize_axes,
    reduction_groups,
    reduction_shape,
)
from ._normalization import shifted_normalization


def _group_value(a: Tensor, indices: list[int]) -> float:
    """Return ``log(sum(exp(a[indices])))`` without avoidable overflow."""
    if not indices:
        raise ValueError("logsumexp is not defined over an empty reduction")

    if any(math.isnan(float(a._data[index])) for index in indices):
        return math.nan
    group = [float(a._data[index]) for index in indices]
    maximum = max(group)
    if maximum == math.inf:
        return math.inf
    if maximum == -math.inf:
        return -math.inf
    _, correction, _, _ = shifted_normalization(group)
    return maximum + correction


class LogSumExp(Operation):
    """Reduce values with a stable logarithm of summed exponentials."""

    __slots__ = ("axis", "keepdims")
    name = "logsumexp"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        storage = execute_logsumexp(
            a,
            axes,
            keepdims=keepdims,
            dtype=dtype,
            output_shape=output_shape,
        )
        if storage is not None:
            return Tensor._from_owned_storage(storage, dtype=dtype, shape=output_shape)
        _, output_shape, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
        )
        values = [_group_value(a, group) for group in groups]
        return Tensor(values, dtype=dtype, shape=output_shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        axis = self.axis
        keepdims = self.keepdims
        axes = normalize_axes(a.ndim, axis)
        output_shape = reduction_shape(a.shape, axes, keepdims)
        if axis is None and not keepdims:
            output_shape = (1,)
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        storage = execute_logsumexp_gradient(
            grad,
            a,
            axes,
            keepdims=keepdims,
        )
        if storage is not None:
            return [Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=a.shape)]
        _, _, groups = reduction_groups(
            a, axis, keepdims, scalar_as_vector=True
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

            _, _, probabilities, _ = shifted_normalization(
                [float(a._data[index]) for index in group]
            )
            for input_index, probability in zip(group, probabilities):
                values[input_index] = (
                    grad._data[output_index] * probability
                )

        return [Tensor(values, dtype=grad.dtype, shape=a.shape)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable log-sum-exp VJP."""
        from ..variable import Variable

        axis = self.axis
        keepdims = self.keepdims
        value = inputs[0]
        operation = LogSumExpGradient(axis=axis, keepdims=keepdims)
        return [
            Variable._record_operation(
                operation.forward(grad.data, value.data),
                operation,
                (grad, value),
            )
        ]


class LogSumExpGradient(Operation):
    """Differentiable, infinity-safe VJP used by :class:`LogSumExp`."""

    __slots__ = ("axis", "keepdims")
    name = "logsumexp_gradient"

    def __init__(
        self,
        *,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "keepdims", keepdims)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        axis = self.axis
        keepdims = self.keepdims
        return LogSumExp(axis=axis, keepdims=keepdims).backward(
            grad,
            value,
            needs_input_grad=UNARY_DEMAND,
        )[0]

    def backward(
        self,
        outer_grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        from .sum import _stable_product_sum

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        keepdims = self.keepdims
        _, output_shape, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        grad_values = [0.0] * grad.size
        value_values = [0.0] * value.size

        for output_index, group in enumerate(groups):
            group_values = [float(value._data[index]) for index in group]
            maximum = max(group_values)
            at_positive_infinity = maximum == math.inf
            if any(math.isnan(item) for item in group_values):
                weights = [math.nan] * len(group)
            elif at_positive_infinity:
                count = sum(item == math.inf for item in group_values)
                weights = [
                    1.0 / count if item == math.inf else 0.0
                    for item in group_values
                ]
            elif maximum == -math.inf:
                raise ValueError(
                    "logsumexp gradient is undefined when every reduced value is -inf"
                )
            else:
                _, _, weights, _ = shifted_normalization(group_values)

            projection = _stable_product_sum([
                (float(outer_grad._data[index]), weight)
                for index, weight in zip(group, weights)
            ])
            grad_values[output_index] = projection
            if at_positive_infinity or not need_value:
                continue
            group_grad = float(grad._data[output_index])
            for index, weight in zip(group, weights):
                value_values[index] = (
                    group_grad
                    * weight
                    * (float(outer_grad._data[index]) - projection)
                )

        return [
            Tensor(grad_values, dtype=outer_grad.dtype, shape=output_shape)
            if need_grad
            else None,
            Tensor(value_values, dtype=outer_grad.dtype, shape=value.shape)
            if need_value
            else None,
        ]

    def backward_graph(
        self,
        outer_grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a smooth third-order rule away from infinite inputs."""
        from ..variable import Variable
        from .exp import exp
        from .reshape import reshape
        from .sum import sum

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        keepdims = self.keepdims
        _, _, groups = reduction_groups(
            value.data, axis, keepdims, scalar_as_vector=True
        )
        if any(
            any(math.isinf(float(value.data._data[index])) for index in group)
            for group in groups
        ):
            numerical = LogSumExpGradient(
                axis=axis,
                keepdims=keepdims,
            ).backward(
                outer_grad.data,
                grad.data,
                value.data,
                needs_input_grad=needs_input_grad,
            )
            return [
                Variable(result, requires_grad=False)
                if result is not None
                else None
                for result in numerical
            ]

        normalizer = logsumexp(value, axis=axis, keepdims=True)
        weights = exp(value - normalizer)
        grad_gradient = None
        if need_grad:
            grad_gradient = sum(
                outer_grad * weights,
                axis=axis,
                keepdims=keepdims,
            )
        value_gradient = None
        if need_value:
            projection = sum(outer_grad * weights, axis=axis, keepdims=True)
            expanded_grad = (
                grad
                if keepdims
                else reshape(grad, keepdims_shape(value.shape, axis))
            )
            value_gradient = expanded_grad * weights * (outer_grad - projection)
        return [grad_gradient, value_gradient]


@overload
def logsumexp(
    value: TensorValue,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorValue: ...


@overload
def logsumexp(
    value: TensorData,
    axis: Axis = None,
    keepdims: bool = False,
) -> Tensor: ...


def logsumexp(
    value: TensorLike,
    axis: Axis = None,
    keepdims: bool = False,
) -> TensorResult:
    """Compute ``log(sum(exp(value)))`` stably over selected axes."""
    from ..variable import Variable

    if isinstance(axis, list):
        axis = tuple(axis)

    if isinstance(value, Variable):
        operation = LogSumExp(axis=axis, keepdims=keepdims)
        return Variable._record_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return LogSumExp(axis=axis, keepdims=keepdims).forward(value)


__all__ = ["LogSumExp", "logsumexp"]
