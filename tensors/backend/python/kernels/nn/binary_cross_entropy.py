"""Reference binary cross-entropy for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.summation import stable_float_mean
from tensors.utils.summation import stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _probability_loss(probability: float, target: float) -> float:
    """Return the loss for a probability input, exact at the boundaries."""
    if probability == 0.0:
        return 0.0 if target == 0.0 else math.inf
    if probability == 1.0:
        return 0.0 if target == 1.0 else math.inf
    return -target * math.log(probability) - (1.0 - target) * math.log1p(-probability)


def _logit_loss(value: float, target: float) -> float:
    """Return the loss for a logit input without overflowing ``exp``."""
    if value == math.inf:
        return 0.0 if target == 1.0 else math.inf
    if value == -math.inf:
        return 0.0 if target == 0.0 else math.inf
    if value >= 0.0:
        return (1.0 - target) * value + math.log1p(math.exp(-value))
    return -target * value + math.log1p(math.exp(value))


def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Compute the elementwise losses, then apply ``reduction``."""
    values = []
    for raw_prediction, raw_target in zip(prediction._data, target._data):
        value = float(raw_prediction)
        target_value = float(raw_target)
        if from_logits:
            values.append(_logit_loss(value, target_value))
            continue
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        values.append(_probability_loss(value, target_value))
    if reduction == "none":
        return PythonStorage.from_values(values, dtype)
    if reduction == "mean":
        total = stable_float_mean(values)
    else:
        total = stable_float_sum(values)
    return PythonStorage.from_values([total], dtype)
