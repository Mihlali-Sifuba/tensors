"""Reference multiclass cross-entropy VJP for the Python backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend.python.kernels.nn._normalization import (
    _axis_positions,
    _floating_dtype,
    _normalization_components,
)
from tensors.backend.python.kernels.nn.log_softmax import log_softmax
from tensors.backend.python.storage import PythonStorage
from tensors.utils.summation import stable_float_sum

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def cross_entropy_gradient(
    grad_values: Sequence[Any],
    logits_values: Sequence[Any],
    target_values: Sequence[Any],
    grad_shape: tuple[int, ...],
    logits_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    logits_dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Return cancellation-resistant VJPs from backend-native values."""
    if logits_shape != target_shape:
        return None
    sample_shape = logits_shape[:axis] + logits_shape[axis + 1 :]
    output_size = math.prod(sample_shape)
    if reduction == "none":
        if grad_shape != sample_shape and not (
            sample_shape == () and grad_shape == (1,)
        ):
            return None
        upstream_values = list(grad_values)
    else:
        if math.prod(grad_shape) != 1:
            return None
        scale = 1.0 / output_size if reduction == "mean" and output_size else 1.0
        upstream_values = [grad_values[0] * scale] * output_size
    need_logits, need_targets = needs_input_grad
    probabilities: list[float] = []
    complements: list[float] = []
    if need_logits:
        probabilities, complements = _normalization_components(
            logits_values, logits_shape, axis, logits_dtype
        )
    log_probabilities: Sequence[Any] = []
    if need_targets:
        log_probabilities = log_softmax(
            logits_values,
            logits_shape,
            axis,
            dtype=_floating_dtype(logits_dtype),
        ).buffer
    logits_gradient = [0.0] * math.prod(logits_shape)
    targets_gradient = [0.0] * math.prod(target_shape)
    for output_index, group in enumerate(_axis_positions(logits_shape, axis)):
        upstream = upstream_values[output_index]
        if upstream == 0:
            continue
        if need_targets:
            for index in group:
                targets_gradient[index] = -upstream * float(log_probabilities[index])
        if not need_logits:
            continue
        target_mass = math.fsum(float(target_values[index]) for index in group)
        for index in group:
            probability = probabilities[index]
            target = float(target_values[index])
            if probability > 0.5:
                derivative = stable_float_sum(
                    [target_mass - target, -target_mass * complements[index]]
                )
            else:
                derivative = target_mass * probability - target
            logits_gradient[index] = upstream * derivative
    return (
        PythonStorage.from_values(logits_gradient, dtype) if need_logits else None,
        PythonStorage.from_values(targets_gradient, dtype) if need_targets else None,
    )
