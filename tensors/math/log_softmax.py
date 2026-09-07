"""Numerically stable log-softmax and its differentiation rule."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, overload

import math

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_normalization, execute_normalization_gradient
from ..dtype import float64
from ..ops.operation import Operation
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._normalization import shifted_normalization
from .softmax import (
    Softmax,
    _axis_layout,
    _centered_softmax_tensor,
    _normalization_components,
    _normalize_axis,
    _softmax_centered,
    _softmax_vjp,
    _softmax_vjp_tensor,
)

if TYPE_CHECKING:
    from ..graph.node import VariableNode


class LogSoftmax(Operation):
    """Normalize logits in log space along one axis."""

    __slots__ = ("axis",)
    name = "log_softmax"

    def __init__(
        self,
        *,
        axis: int = -1,
    ) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, a: Tensor) -> Tensor:
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        storage = execute_normalization("log_softmax", a, axis, dtype=dtype)
        if storage is not None:
            return Tensor._from_owned_storage(storage, dtype=dtype, shape=a.shape)
        values = [0.0] * a.size

        for group in range(before):
            group_start = group * axis_size * trailing
            for offset in range(trailing):
                positions = [
                    group_start + offset + index * trailing
                    for index in range(axis_size)
                ]
                if any(
                    math.isnan(float(a._data[position]))
                    for position in positions
                ):
                    for position in positions:
                        values[position] = math.nan
                    continue
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
                _, correction, _, _ = shifted_normalization(
                    [float(a._data[position]) for position in positions]
                )
                for position in positions:
                    values[position] = (
                        float(a._data[position]) - maximum - correction
                    )

        return Tensor(values, dtype=dtype, shape=a.shape)

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        a = inputs[0]
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        return [_log_softmax_vjp_tensor(grad, a, axis)]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable log-softmax VJP."""
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(inputs[0].data, axis)
        return [_log_softmax_vjp(grad, inputs[0], axis)]


def _log_softmax_vjp_tensor(
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Tensor:
    """Return a cancellation-resistant log-softmax VJP."""
    storage = execute_normalization_gradient(
        "log_softmax",
        grad,
        value,
        axis,
    )
    if storage is not None:
        return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=value.shape)
    from .sum import _stable_product_sum

    probabilities, complements = _normalization_components(value, axis)
    before, axis_size, trailing = _axis_layout(value, axis)
    values = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing
                for index in range(axis_size)
            ]
            for position in positions:
                probability = float(probabilities._data[position])
                terms = [
                    (float(grad._data[position]), complements[position])
                ]
                terms.extend(
                    (-float(grad._data[other]), probability)
                    for other in positions
                    if other != position
                )
                values[position] = _stable_product_sum(terms)
    return Tensor(values, dtype=grad.dtype, shape=value.shape)


class LogSoftmaxGradient(Operation):
    """Differentiable, cancellation-resistant log-softmax VJP."""

    __slots__ = ("axis",)
    name = "log_softmax_gradient"

    def __init__(
        self,
        *,
        axis: int,
    ) -> None:
        object.__setattr__(self, "axis", axis)

    def forward(self, grad: Tensor, value: Tensor) -> Tensor:
        axis = self.axis
        return _log_softmax_vjp_tensor(grad, value, axis)

    def backward(
        self,
        outer_grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Tensor]:
        from .sum import Sum
        from ..utils.broadcasting import broadcast_to

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        assert isinstance(axis, int)
        value_gradient = None
        if need_value:
            total = Sum(axis=axis, keepdims=True).forward(grad)
            expanded_total = broadcast_to(total, value.shape)
            value_vjp = _softmax_vjp_tensor(outer_grad, value, axis)
            value_gradient = Tensor(
                [
                    -scale * derivative
                    for scale, derivative in zip(
                        expanded_total._data,
                        value_vjp._data,
                    )
                ],
                dtype=outer_grad.dtype,
                shape=value.shape,
            )
        return [
            _centered_softmax_tensor(outer_grad, value, axis)
            if need_grad
            else None,
            value_gradient,
        ]

    def backward_graph(
        self,
        outer_grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from .sum import sum

        grad, value = inputs
        need_grad, need_value = needs_input_grad
        axis = self.axis
        return [
            _softmax_centered(outer_grad, value, axis) if need_grad else None,
            -sum(grad, axis=axis, keepdims=True)
            * _softmax_vjp(outer_grad, value, axis)
            if need_value
            else None,
        ]


def _log_softmax_vjp(grad, value, axis: int):
    from ..variable import Variable

    operation = LogSoftmaxGradient(axis=axis)
    return Variable._apply_operation(operation, (grad, value))


@overload
def log_softmax(value: VariableNode, axis: int = -1) -> VariableNode: ...


@overload
def log_softmax(value: TensorValue, axis: int = -1) -> TensorValue: ...


@overload
def log_softmax(value: TensorData, axis: int = -1) -> Tensor: ...


def log_softmax(
    value: TensorLike | VariableNode,
    axis: int = -1,
) -> TensorResult | VariableNode:
    """Return stable log probabilities along ``axis``.

    A graph value is applied through the graph: a Variable calculates the
    result now, and a vertex records the operation for a program that runs
    later. The configuration the operation is built with is the one every
    application uses, so a recorded call replays what it was written as.
    """
    from ..graph.expression import apply_operation, is_graph_operand

    operation = LogSoftmax(axis=axis)

    if is_graph_operand(value):
        return apply_operation(operation, (value,))
    return operation.forward(as_tensor_operand(value))


__all__ = ["LogSoftmax", "log_softmax"]
