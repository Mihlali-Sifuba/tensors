"""Reference multiclass cross-entropy VJP for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.kernels.nn._normalization import (
    _log_softmax_values,
    _softmax_values,
)
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.normalization import shifted_normalization
from tensors.utils.reductions import reduction_groups
from tensors.utils.summation import stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.tensor import Tensor


def cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None]:
    """Return the requested logit and target gradients.

    The logit derivative ``target_mass * p - target`` cancels badly once a
    probability approaches one, so above one half it is accumulated from the
    probability's complement instead.
    """
    need_logits, need_targets = needs_input_grad
    probabilities = _softmax_values(logits, axis) if need_logits else []
    log_probabilities = _log_softmax_values(logits, axis) if need_targets else []
    _, _, groups = reduction_groups(logits.shape, axis, keepdims=False)
    logits_gradient = [0.0] * logits.size
    targets_gradient = [0.0] * targets.size
    for output_index, group in enumerate(groups):
        if reduction == "none":
            upstream = grad._data[output_index]
        else:
            upstream = grad._data[0]
            if reduction == "mean" and groups:
                upstream /= len(groups)
        if upstream == 0:
            continue
        if need_targets:
            for index in group:
                targets_gradient[index] = -upstream * log_probabilities[index]
        if not need_logits:
            continue
        target_mass = math.fsum(float(targets._data[index]) for index in group)
        group_values = [float(logits._data[index]) for index in group]
        if group_values and all(math.isfinite(value) for value in group_values):
            _, _, _, complements = shifted_normalization(group_values)
        else:
            complements = [1.0 - probabilities[index] for index in group]
        for index, complement in zip(group, complements):
            probability = probabilities[index]
            target_value = float(targets._data[index])
            if probability > 0.5:
                derivative = stable_float_sum(
                    [target_mass - target_value, -target_mass * complement]
                )
            else:
                derivative = target_mass * probability - target_value
            logits_gradient[index] = upstream * derivative
    return (
        PythonStorage.from_values(logits_gradient, grad.dtype) if need_logits else None,
        (
            PythonStorage.from_values(targets_gradient, grad.dtype)
            if need_targets
            else None
        ),
    )
