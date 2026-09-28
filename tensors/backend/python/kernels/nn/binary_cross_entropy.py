"""Reference binary cross-entropy for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.storage import PythonStorage
from tensors.utils.summation import stable_float_mean, stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def _probability_loss(probability: float, target: float) -> float:
    if probability == 0.0:
        return 0.0 if target == 0.0 else math.inf
    if probability == 1.0:
        return 0.0 if target == 1.0 else math.inf
    return -target * math.log(probability) - (1.0 - target) * math.log1p(-probability)


def _logit_loss(value: float, target: float) -> float:
    if value == math.inf:
        return 0.0 if target == 1.0 else math.inf
    if value == -math.inf:
        return 0.0 if target == 0.0 else math.inf
    if value >= 0.0:
        return (1.0 - target) * value + math.log1p(math.exp(-value))
    return -target * value + math.log1p(math.exp(value))


def binary_cross_entropy(
    prediction_values: Sequence[Any],
    target_values: Sequence[Any],
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> PythonStorage | None:
    """Compute elementwise losses from native values, then reduce them."""
    if prediction_shape != target_shape:
        return None
    values = []
    for raw_prediction, raw_target in zip(prediction_values, target_values):
        value = float(raw_prediction)
        target = float(raw_target)
        if from_logits:
            values.append(_logit_loss(value, target))
            continue
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        values.append(_probability_loss(value, target))
    if reduction == "none":
        result = values
    elif reduction == "mean":
        result = [stable_float_mean(values)]
    else:
        result = [stable_float_sum(values)]
    storage = PythonStorage.from_values(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Python loss kernel returned an unexpected result size")
    return storage
