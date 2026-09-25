"""CUDA-native binary cross-entropy VJP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def binary_cross_entropy_gradient(
    grad_values: Any,
    prediction_values: Any,
    target_values: Any,
    grad_shape: tuple[int, ...],
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Compute requested VJPs from device-native values."""
    if prediction_shape != target_shape:
        return None
    values = _widen(cupy.asarray(prediction_values).reshape(prediction_shape))
    targets = _widen(cupy.asarray(target_values).reshape(target_shape))
    upstream = _widen(cupy.asarray(grad_values).reshape(grad_shape))
    if reduction == "none":
        if grad_shape != prediction_shape:
            return None
        expanded_upstream = upstream
    else:
        if upstream.size != 1:
            return None
        scale = 1.0 / values.size if reduction == "mean" and values.size else 1.0
        expanded_upstream = (
            cupy.broadcast_to(upstream.reshape(()), prediction_shape) * scale
        )
    need_prediction, need_target = needs_input_grad
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if from_logits:
            magnitude = cupy.exp(-cupy.abs(values))
            sigmoid = cupy.where(
                values >= 0.0,
                1.0 / (1.0 + magnitude),
                magnitude / (1.0 + magnitude),
            )
            prediction_derivative = cupy.where(
                values >= 0.0,
                1.0 - targets - magnitude / (1.0 + magnitude),
                sigmoid - targets,
            )
            target_derivative = -values
        else:
            invalid = ~cupy.isfinite(values) | (values < 0.0) | (values > 1.0)
            if bool(cupy.any(invalid)):
                raise ValueError(
                    "binary cross-entropy probabilities must be between 0 and 1"
                )
            prediction_derivative = cupy.where(
                values == 0.0,
                cupy.where(targets == 0.0, 1.0, -cupy.inf),
                cupy.where(
                    values == 1.0,
                    cupy.where(targets == 1.0, -1.0, cupy.inf),
                    (values - targets) / (values * (1.0 - values)),
                ),
            )
            target_derivative = cupy.where(
                values == 0.0,
                cupy.inf,
                cupy.where(
                    values == 1.0,
                    -cupy.inf,
                    cupy.log1p(-values) - cupy.log(values),
                ),
            )
        zero_upstream = expanded_upstream == 0.0
        prediction_result = cupy.where(
            zero_upstream, 0.0, expanded_upstream * prediction_derivative
        )
        target_result = cupy.where(
            zero_upstream, 0.0, expanded_upstream * target_derivative
        )
    prediction_storage = (
        _storage(prediction_result, dtype=dtype, output_shape=prediction_shape)
        if need_prediction
        else None
    )
    target_storage = (
        _storage(target_result, dtype=dtype, output_shape=target_shape)
        if need_target
        else None
    )
    if (need_prediction and prediction_storage is None) or (
        need_target and target_storage is None
    ):
        return None
    return prediction_storage, target_storage
