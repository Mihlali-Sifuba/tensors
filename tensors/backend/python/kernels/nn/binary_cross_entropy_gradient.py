"""Reference binary cross-entropy VJP for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.elementwise.sigmoid import _sigmoid
from tensors.backend.python.storage import PythonStorage

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def _probability_gradient(probability: float, target: float) -> float:
    if probability == 0.0:
        return 1.0 if target == 0.0 else -math.inf
    if probability == 1.0:
        return -1.0 if target == 1.0 else math.inf
    return (probability - target) / (probability * (1.0 - probability))


def _logit_gradient(value: float, target: float) -> float:
    if value >= 0.0:
        return 1.0 - target - _sigmoid(-value)
    return _sigmoid(value) - target


def _target_gradient(probability: float) -> float:
    if probability == 0.0:
        return math.inf
    if probability == 1.0:
        return -math.inf
    return math.log1p(-probability) - math.log(probability)


def binary_cross_entropy_gradient(
    grad_values: Sequence[Any],
    prediction_values: Sequence[Any],
    target_values: Sequence[Any],
    grad_shape: tuple[int, ...],
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Return requested VJPs from backend-native values."""
    if prediction_shape != target_shape:
        return None
    size = math.prod(prediction_shape)
    if reduction == "none":
        if grad_shape != prediction_shape:
            return None
        upstream = list(grad_values)
    else:
        if math.prod(grad_shape) != 1:
            return None
        scale = 1.0 / size if reduction == "mean" and size else 1.0
        upstream = [grad_values[0] * scale] * size
    need_prediction, need_target = needs_input_grad
    prediction_gradients = []
    target_gradients = []
    for upstream_value, raw_prediction, raw_target in zip(
        upstream, prediction_values, target_values
    ):
        if upstream_value == 0:
            if need_prediction:
                prediction_gradients.append(0.0)
            if need_target:
                target_gradients.append(0.0)
            continue
        value = float(raw_prediction)
        target = float(raw_target)
        if need_prediction:
            derivative = (
                _logit_gradient(value, target)
                if from_logits
                else _probability_gradient(value, target)
            )
            prediction_gradients.append(upstream_value * derivative)
        if need_target:
            target_gradients.append(
                upstream_value * (-value if from_logits else _target_gradient(value))
            )
    return (
        (
            PythonStorage.from_values(prediction_gradients, dtype)
            if need_prediction
            else None
        ),
        PythonStorage.from_values(target_gradients, dtype) if need_target else None,
    )
