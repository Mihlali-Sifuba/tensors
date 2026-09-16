"""Reference binary cross-entropy VJP for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.math.sigmoid import _sigmoid

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.tensor import Tensor


def _probability_gradient(probability: float, target: float) -> float:
    """Return the prediction derivative for a probability input."""
    if probability == 0.0:
        return 1.0 if target == 0.0 else -math.inf
    if probability == 1.0:
        return -1.0 if target == 1.0 else math.inf
    return (probability - target) / (probability * (1.0 - probability))


def _logit_gradient(value: float, target: float) -> float:
    """Return the prediction derivative for a logit input.

    ``sigmoid(x) - t`` is evaluated from whichever side keeps the subtraction
    away from a saturated sigmoid.
    """
    if value >= 0.0:
        return 1.0 - target - _sigmoid(-value)
    return _sigmoid(value) - target


def _target_gradient(probability: float) -> float:
    """Return the target derivative for a probability input."""
    if probability == 0.0:
        return math.inf
    if probability == 1.0:
        return -math.inf
    return math.log1p(-probability) - math.log(probability)


def binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Return the requested prediction and target gradients."""
    need_prediction, need_target = needs_input_grad
    size = prediction.size
    if reduction == "none":
        upstream = list(grad._data)
    else:
        scale = 1.0 / size if reduction == "mean" and size else 1.0
        upstream = [grad._data[0] * scale] * size
    prediction_gradients = []
    target_gradients = []
    for upstream_value, raw_prediction, raw_target in zip(
        upstream,
        prediction._data,
        target._data,
    ):
        if upstream_value == 0:
            if need_prediction:
                prediction_gradients.append(0.0)
            if need_target:
                target_gradients.append(0.0)
            continue
        value = float(raw_prediction)
        target_value = float(raw_target)
        if need_prediction:
            derivative = (
                _logit_gradient(value, target_value)
                if from_logits
                else _probability_gradient(value, target_value)
            )
            prediction_gradients.append(upstream_value * derivative)
        if need_target:
            target_gradients.append(
                upstream_value * (-value if from_logits else _target_gradient(value))
            )
    return (
        (
            PythonStorage.from_values(prediction_gradients, grad.dtype)
            if need_prediction
            else None
        ),
        (
            PythonStorage.from_values(target_gradients, grad.dtype)
            if need_target
            else None
        ),
    )
