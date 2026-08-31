"""Numerically stable log-softmax and its differentiation rule."""

from __future__ import annotations

from typing import Any, List, overload

import math

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_normalization, execute_normalization_gradient
from ..dtype import float64
from ..tensor import Tensor
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


class LogSoftmax:
    """Normalize logits in log space along one axis."""

    @staticmethod
    def forward(a: Tensor, axis: int = -1) -> Tensor:
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

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a = inputs[0]
        axis = kwargs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("log_softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        return [_log_softmax_vjp_tensor(grad, a, axis)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable log-softmax VJP."""
        axis = kwargs.get("axis", -1)
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


class LogSoftmaxGradient:
    """Differentiable, cancellation-resistant log-softmax VJP."""

    @staticmethod
    def forward(grad: Tensor, value: Tensor, *, axis: int) -> Tensor:
        return _log_softmax_vjp_tensor(grad, value, axis)

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        from .sum import Sum
        from ..utils.broadcasting import broadcast_to

        grad, value = inputs
        axis = kwargs["axis"]
        assert isinstance(axis, int)
        total = Sum.forward(grad, axis=axis, keepdims=True)
        expanded_total = broadcast_to(total, value.shape)
        value_vjp = _softmax_vjp_tensor(outer_grad, value, axis)
        return [
            _centered_softmax_tensor(outer_grad, value, axis),
            Tensor(
                [
                    -scale * derivative
                    for scale, derivative in zip(
                        expanded_total._data,
                        value_vjp._data,
                    )
                ],
                dtype=outer_grad.dtype,
                shape=value.shape,
            ),
        ]

    @staticmethod
    def backward_graph(outer_grad, *inputs, **kwargs: object):
        from .sum import sum

        grad, value = inputs
        axis = kwargs["axis"]
        return [
            _softmax_centered(outer_grad, value, axis),
            -sum(grad, axis=axis, keepdims=True)
            * _softmax_vjp(outer_grad, value, axis),
        ]


def _log_softmax_vjp(grad, value, axis: int):
    from ..variable import Variable

    return Variable._from_operation(
        LogSoftmaxGradient.forward(grad.data, value.data, axis=axis),
        "log_softmax_gradient",
        LogSoftmaxGradient,
        [grad, value],
        axis=axis,
    )


@overload
def log_softmax(value: TensorValue, axis: int = -1) -> TensorValue: ...


@overload
def log_softmax(value: TensorData, axis: int = -1) -> Tensor: ...


def log_softmax(value: TensorLike, axis: int = -1) -> TensorResult:
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
