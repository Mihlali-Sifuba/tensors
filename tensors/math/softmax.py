"""Numerically stable softmax and its differentiation rule."""

from __future__ import annotations

import math
from typing import Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_normalization, execute_normalization_gradient
from ..dtype import float64
from ..tensor import Tensor
from ._normalization import shifted_normalization


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
        if not isinstance(keepdims, bool):
            raise TypeError("keepdims must be a bool")
        if keepdims:
            raise ValueError("softmax does not support keepdims")
        axis = _normalize_axis(a, axis)
        before, axis_size, trailing = _axis_layout(a, axis)
        dtype = a.dtype if a.dtype.typecode in {"f", "d"} else float64
        storage = execute_normalization("softmax", a, axis, dtype=dtype)
        if storage is not None:
            return Tensor._from_owned_storage(storage, dtype=dtype, shape=a.shape)
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
                _, _, probabilities, _ = shifted_normalization(
                    [float(a._data[position]) for position in positions]
                )
                for position, probability in zip(positions, probabilities):
                    values[position] = probability

        return Tensor(values, dtype=dtype, shape=a.shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Apply the softmax Jacobian-vector product along ``axis``."""
        a = inputs[0]
        axis = kwargs.get("axis", -1)
        if not isinstance(axis, int):
            raise TypeError("softmax axis must be an integer")
        axis = _normalize_axis(a, axis)
        return [_softmax_vjp_tensor(grad, a, axis)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build the differentiable softmax Jacobian-vector product."""
        axis = kwargs.get("axis", -1)
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("softmax axis must be an integer")
        axis = _normalize_axis(inputs[0].data, axis)
        return [_softmax_vjp(grad, inputs[0], axis)]


def _normalization_components(
    value: Tensor,
    axis: int,
) -> tuple[Tensor, list[float]]:
    """Return probabilities and accurately represented complements."""
    probabilities = Softmax.forward(value, axis=axis)
    before, axis_size, trailing = _axis_layout(value, axis)
    complements = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing
                for index in range(axis_size)
            ]
            group_values = [float(value._data[position]) for position in positions]
            if all(math.isfinite(item) for item in group_values):
                _, _, _, group_complements = shifted_normalization(group_values)
            else:
                group_complements = [
                    1.0 - float(probabilities._data[position])
                    for position in positions
                ]
            for position, complement in zip(positions, group_complements):
                complements[position] = complement
    return probabilities, complements


def _centered_softmax_tensor(
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Tensor:
    """Return ``grad - E_softmax(grad)`` without dominant cancellation."""
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
                terms = [
                    (float(grad._data[position]), complements[position])
                ]
                terms.extend(
                    (
                        -float(grad._data[other]),
                        float(probabilities._data[other]),
                    )
                    for other in positions
                    if other != position
                )
                values[position] = _stable_product_sum(terms)
    return Tensor(values, dtype=grad.dtype, shape=value.shape)


def _softmax_vjp_tensor(grad: Tensor, value: Tensor, axis: int) -> Tensor:
    """Return a cancellation-resistant softmax Jacobian-vector product."""
    storage = execute_normalization_gradient("softmax", grad, value, axis)
    if storage is not None:
        return Tensor._from_owned_storage(storage, dtype=grad.dtype, shape=value.shape)
    probabilities = Softmax.forward(value, axis=axis)
    centered = _centered_softmax_tensor(grad, value, axis)
    return Tensor(
        [
            probability * difference
            for probability, difference in zip(
                probabilities._data,
                centered._data,
            )
        ],
        dtype=grad.dtype,
        shape=value.shape,
    )


def _softmax_expectation_tensor(
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Tensor:
    """Broadcast the softmax-weighted expectation of ``grad`` per group."""
    from .sum import _stable_product_sum

    probabilities = Softmax.forward(value, axis=axis)
    before, axis_size, trailing = _axis_layout(value, axis)
    values = [0.0] * value.size
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            positions = [
                group_start + offset + index * trailing
                for index in range(axis_size)
            ]
            expectation = _stable_product_sum([
                (
                    float(grad._data[position]),
                    float(probabilities._data[position]),
                )
                for position in positions
            ])
            for position in positions:
                values[position] = expectation
    return Tensor(values, dtype=grad.dtype, shape=value.shape)


class SoftmaxCentered:
    """Differentiable softmax-expectation centering operation."""

    @staticmethod
    def forward(grad: Tensor, value: Tensor, *, axis: int) -> Tensor:
        return _centered_softmax_tensor(grad, value, axis)

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        from .log_softmax import _log_softmax_vjp_tensor
        from .sum import Sum

        grad, value = inputs
        axis = kwargs["axis"]
        assert isinstance(axis, int)
        total = Sum.forward(outer_grad, axis=axis, keepdims=True)
        expanded_total = _broadcast_reduction(total, value)
        value_vjp = _softmax_vjp_tensor(grad, value, axis)
        return [
            _log_softmax_vjp_tensor(outer_grad, value, axis),
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
        from .log_softmax import _log_softmax_vjp
        from .sum import sum

        grad, value = inputs
        axis = kwargs["axis"]
        return [
            _log_softmax_vjp(outer_grad, value, axis),
            -sum(outer_grad, axis=axis, keepdims=True)
            * _softmax_vjp(grad, value, axis),
        ]


class SoftmaxGradient:
    """Differentiable, cancellation-resistant softmax VJP."""

    @staticmethod
    def forward(grad: Tensor, value: Tensor, *, axis: int) -> Tensor:
        return _softmax_vjp_tensor(grad, value, axis)

    @staticmethod
    def backward(
        outer_grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        from .sum import _stable_product_sum

        grad, value = inputs
        axis = kwargs["axis"]
        assert isinstance(axis, int)
        centered = _centered_softmax_tensor(grad, value, axis)
        projections = _softmax_expectation_tensor(outer_grad, value, axis)
        vector = Tensor(
            [
                _stable_product_sum([
                    (float(outer), float(difference)),
                    (-float(item), projection),
                ])
                for outer, difference, item, projection in zip(
                    outer_grad._data,
                    centered._data,
                    grad._data,
                    projections._data,
                )
            ],
            dtype=outer_grad.dtype,
            shape=value.shape,
        )
        return [
            _softmax_vjp_tensor(outer_grad, value, axis),
            _softmax_vjp_tensor(vector, value, axis),
        ]

    @staticmethod
    def backward_graph(outer_grad, *inputs, **kwargs: object):
        from .sum import sum

        grad, value = inputs
        axis = kwargs["axis"]
        centered = _softmax_centered(grad, value, axis)
        projection = sum(
            outer_grad * softmax(value, axis=axis),
            axis=axis,
            keepdims=True,
        )
        vector = outer_grad * centered - grad * projection
        return [
            _softmax_vjp(outer_grad, value, axis),
            _softmax_vjp(vector, value, axis),
        ]


def _broadcast_reduction(reduced: Tensor, value: Tensor) -> Tensor:
    from ..utils.broadcasting import broadcast_to

    return broadcast_to(reduced, value.shape)


def _softmax_centered(grad, value, axis: int):
    from ..variable import Variable

    return Variable._from_operation(
        SoftmaxCentered.forward(grad.data, value.data, axis=axis),
        "softmax_centered",
        SoftmaxCentered,
        [grad, value],
        axis=axis,
    )


def _softmax_vjp(grad, value, axis: int):
    from ..variable import Variable

    return Variable._from_operation(
        SoftmaxGradient.forward(grad.data, value.data, axis=axis),
        "softmax_gradient",
        SoftmaxGradient,
        [grad, value],
        axis=axis,
    )


@overload
def softmax(value: TensorValue, axis: int = -1) -> TensorValue: ...


@overload
def softmax(value: TensorData, axis: int = -1) -> Tensor: ...


def softmax(value: TensorLike, axis: int = -1) -> TensorResult:
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
