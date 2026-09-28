"""CUDA-native binary cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate, _storage, _widen
from tensors.backend.cuda.kernels.nn.losses import _reduce_losses
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType


def binary_cross_entropy(
    prediction_values: Any,
    target_values: Any,
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Compute binary cross-entropy from device-native values."""
    if prediction_shape != target_shape:
        return None
    values = _widen(cupy.asarray(prediction_values).reshape(prediction_shape))
    targets = _widen(cupy.asarray(target_values).reshape(target_shape))
    if from_logits:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            correction = cupy.log1p(cupy.exp(-cupy.abs(values)))
            ordinary = cupy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
            losses = cupy.where(
                cupy.isposinf(values),
                cupy.where(targets == 1.0, 0.0, cupy.inf),
                cupy.where(
                    cupy.isneginf(values),
                    cupy.where(targets == 0.0, 0.0, cupy.inf),
                    ordinary,
                ),
            )
    else:
        invalid = ~cupy.isfinite(values) | (values < 0.0) | (values > 1.0)
        if bool(cupy.any(invalid)):
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(divide="ignore", invalid="ignore"):
            losses = cupy.where(
                values == 0.0,
                cupy.where(targets == 0.0, 0.0, cupy.inf),
                cupy.where(
                    values == 1.0,
                    cupy.where(targets == 1.0, 0.0, cupy.inf),
                    -targets * cupy.log(values) - (1.0 - targets) * cupy.log1p(-values),
                ),
            )
    return _storage(
        _reduce_losses(losses, reduction),
        dtype=dtype,
        output_shape=output_shape,
    )
