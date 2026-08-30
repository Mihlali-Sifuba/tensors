"""Numerically stable binary cross-entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import (
    execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient,
)
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape, sum_to_shape_graph
from ..shape import Shape
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_shape, broadcast_tensors
from .cross_entropy import Reduction, _validate_reduction
from .mean import _stable_float_mean
from .sigmoid import _sigmoid
from .sum import _stable_float_sum

if TYPE_CHECKING:
    from ..variable import Variable


def _validate_targets(target: Tensor) -> None:
    if any(not 0.0 <= float(value) <= 1.0 for value in target._data):
        raise ValueError("binary cross-entropy targets must be between 0 and 1")


def _validate_from_logits(from_logits: bool) -> None:
    if not isinstance(from_logits, bool):
        raise TypeError("from_logits must be a bool")


def _probability_loss(probability: float, target: float) -> float:
    if probability == 0.0:
        return 0.0 if target == 0.0 else math.inf
    if probability == 1.0:
        return 0.0 if target == 1.0 else math.inf
    return -target * math.log(probability) - (1.0 - target) * math.log1p(-probability)


def _probability_gradient(probability: float, target: float) -> float:
    if probability == 0.0:
        return 1.0 if target == 0.0 else -math.inf
    if probability == 1.0:
        return -1.0 if target == 1.0 else math.inf
    return (probability - target) / (probability * (1.0 - probability))


def _target_gradient(probability: float) -> float:
    if probability == 0.0:
        return math.inf
    if probability == 1.0:
        return -math.inf
    return math.log1p(-probability) - math.log(probability)


def _reduce(values: list[float], reduction: Reduction) -> tuple[list[float], tuple[int, ...]]:
    if reduction == "none":
        raise RuntimeError("elementwise reduction requires the broadcast shape")
    if reduction == "mean":
        total = _stable_float_mean(values)
    else:
        total = _stable_float_sum(values)
    return [total], (1,)


class BinaryCrossEntropy:
    """Binary cross-entropy for probabilities or raw logits."""

    @staticmethod
    def forward(
        prediction: Tensor,
        target: Tensor,
        *,
        from_logits: bool = False,
        reduction: Reduction = "mean",
    ) -> Tensor:
        _validate_reduction(reduction)
        _validate_from_logits(from_logits)
        prediction, target = broadcast_tensors(prediction, target)
        _validate_targets(target)
        dtype = result_dtype(prediction.dtype, target, division=True)
        output_shape = prediction.shape if reduction == "none" else (1,)
        storage = execute_binary_cross_entropy(
            prediction,
            target,
            from_logits=from_logits,
            reduction=reduction,
            dtype=dtype,
            output_shape=output_shape,
        )
        if storage is not None:
            return Tensor(storage, dtype=dtype, shape=output_shape)

        values = []
        for raw_prediction, raw_target in zip(prediction._data, target._data):
            value = float(raw_prediction)
            target_value = float(raw_target)
            if from_logits:
                if value == math.inf:
                    loss = 0.0 if target_value == 1.0 else math.inf
                elif value == -math.inf:
                    loss = 0.0 if target_value == 0.0 else math.inf
                elif value >= 0.0:
                    loss = (
                        (1.0 - target_value) * value
                        + math.log1p(math.exp(-value))
                    )
                else:
                    loss = -target_value * value + math.log1p(math.exp(value))
            else:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        "binary cross-entropy probabilities must be between 0 and 1"
                    )
                loss = _probability_loss(value, target_value)
            values.append(loss)

        if reduction == "none":
            return Tensor(values, dtype=dtype, shape=prediction.shape)
        reduced, shape = _reduce(values, reduction)
        return Tensor(reduced, dtype=dtype, shape=shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        prediction, target = inputs
        from_logits = kwargs.get("from_logits", False)
        _validate_from_logits(from_logits)
        reduction = kwargs.get("reduction", "mean")
        if not isinstance(reduction, str):
            raise TypeError("reduction must be a string")

        expanded_prediction, expanded_target = broadcast_tensors(prediction, target)
        size = expanded_prediction.size
        accelerated = execute_binary_cross_entropy_gradient(
            grad,
            expanded_prediction,
            expanded_target,
            from_logits=from_logits,
            reduction=reduction,
        )
        if accelerated is not None:
            prediction_storage, target_storage = accelerated
            expanded_shape = expanded_prediction.shape
            return [
                sum_to_shape(
                    Tensor(
                        prediction_storage,
                        dtype=grad.dtype,
                        shape=expanded_shape,
                    ),
                    prediction.shape,
                ),
                sum_to_shape(
                    Tensor(
                        target_storage,
                        dtype=grad.dtype,
                        shape=expanded_shape,
                    ),
                    target.shape,
                ),
            ]
        if reduction == "none":
            upstream = list(grad._data)
        else:
            scale = 1.0 / size if reduction == "mean" and size else 1.0
            upstream = [grad._data[0] * scale] * size

        prediction_gradients = []
        target_gradients = []
        for upstream_value, raw_prediction, raw_target in zip(
            upstream, expanded_prediction._data, expanded_target._data
        ):
            if upstream_value == 0:
                prediction_gradients.append(0.0)
                target_gradients.append(0.0)
                continue
            value = float(raw_prediction)
            target_value = float(raw_target)
            if from_logits:
                prediction_derivative = (
                    (1.0 - target_value) - _sigmoid(-value)
                    if value >= 0.0
                    else _sigmoid(value) - target_value
                )
                target_derivative = -value
            else:
                prediction_derivative = _probability_gradient(value, target_value)
                target_derivative = _target_gradient(value)
            prediction_gradients.append(upstream_value * prediction_derivative)
            target_gradients.append(upstream_value * target_derivative)

        expanded_shape = expanded_prediction.shape
        prediction_gradient = Tensor(
            prediction_gradients, dtype=grad.dtype, shape=expanded_shape
        )
        target_gradient = Tensor(
            target_gradients, dtype=grad.dtype, shape=expanded_shape
        )
        return [
            sum_to_shape(prediction_gradient, prediction.shape),
            sum_to_shape(target_gradient, target.shape),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable binary cross-entropy VJP."""
        from .log import log
        from .sigmoid import sigmoid

        prediction, target = inputs
        from_logits = kwargs.get("from_logits", False)
        _validate_from_logits(from_logits)
        reduction = kwargs.get("reduction", "mean")
        shape = broadcast_shape(prediction.shape, target.shape)
        size = Shape.from_iterable(shape).size
        upstream = grad / size if reduction == "mean" and size else grad

        if from_logits:
            positive_mask = Tensor(
                [
                    1.0 if float(value) >= 0.0 else 0.0
                    for value in prediction.data._data
                ],
                dtype=prediction.dtype,
                shape=prediction.shape,
            )
            negative_mask = Tensor(
                [1.0 - value for value in positive_mask._data],
                dtype=prediction.dtype,
                shape=prediction.shape,
            )
            prediction_derivative = (
                positive_mask * ((1.0 - target) - sigmoid(-prediction))
                + negative_mask * (sigmoid(prediction) - target)
            )
            target_derivative = -prediction + target * 0.0
        else:
            expanded_prediction, expanded_target = broadcast_tensors(
                prediction.data,
                target.data,
            )
            boundary_values = list(
                zip(expanded_prediction._data, expanded_target._data)
            )

            if prediction.requires_grad:
                if any(
                    (value == 0.0 and target_value != 0.0)
                    or (value == 1.0 and target_value != 1.0)
                    for value, target_value in boundary_values
                ):
                    raise ValueError(
                        "Higher-order binary cross-entropy gradients require "
                        "a finite first derivative"
                    )
                zero_mask = Tensor(
                    [
                        1.0 if value == 0.0 and target_value == 0.0 else 0.0
                        for value, target_value in boundary_values
                    ],
                    dtype=prediction.dtype,
                    shape=expanded_prediction.shape,
                )
                one_mask = Tensor(
                    [
                        1.0 if value == 1.0 and target_value == 1.0 else 0.0
                        for value, target_value in boundary_values
                    ],
                    dtype=prediction.dtype,
                    shape=expanded_prediction.shape,
                )
                prediction_derivative = (
                    -target / (prediction + zero_mask)
                    + (1.0 - target) / (1.0 - prediction + one_mask)
                )
            else:
                prediction_derivative = prediction * 0.0

            if target.requires_grad:
                if any(value in {0.0, 1.0} for value, _ in boundary_values):
                    raise ValueError(
                        "Higher-order gradients with respect to binary "
                        "cross-entropy targets require probabilities strictly "
                        "between 0 and 1"
                    )
                target_derivative = (
                    log(1.0 - prediction) - log(prediction) + target * 0.0
                )
            else:
                target_derivative = target * 0.0
        return [
            sum_to_shape_graph(upstream * prediction_derivative, prediction.shape),
            sum_to_shape_graph(upstream * target_derivative, target.shape),
        ]


@overload
def binary_cross_entropy(
    prediction: Variable,
    target: TensorLike,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def binary_cross_entropy(
    prediction: TensorLike,
    target: Variable,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def binary_cross_entropy(
    prediction: TensorData,
    target: TensorData,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> Tensor: ...


def binary_cross_entropy(
    prediction: TensorLike,
    target: TensorLike,
    *,
    from_logits: bool = False,
    reduction: Reduction = "mean",
) -> TensorResult:
    """Compute binary cross-entropy with optional stable logits input."""
    from ..variable import Variable

    prediction_is_variable = isinstance(prediction, Variable)
    target_is_variable = isinstance(target, Variable)
    prediction_tensor = prediction.data if prediction_is_variable else (
        prediction if isinstance(prediction, Tensor) else Tensor(prediction)
    )
    target_tensor = target.data if target_is_variable else (
        target if isinstance(target, Tensor) else Tensor(target)
    )

    if prediction_is_variable or target_is_variable:
        prediction_variable = prediction if prediction_is_variable else Variable(
            prediction_tensor, requires_grad=False
        )
        target_variable = target if target_is_variable else Variable(
            target_tensor, requires_grad=False
        )
        return Variable._from_operation(
            BinaryCrossEntropy.forward(
                prediction_variable.data,
                target_variable.data,
                from_logits=from_logits,
                reduction=reduction,
            ),
            "binary_cross_entropy",
            BinaryCrossEntropy,
            [prediction_variable, target_variable],
            from_logits=from_logits,
            reduction=reduction,
        )
    return BinaryCrossEntropy.forward(
        prediction_tensor,
        target_tensor,
        from_logits=from_logits,
        reduction=reduction,
    )


__all__ = ["BinaryCrossEntropy", "binary_cross_entropy"]
