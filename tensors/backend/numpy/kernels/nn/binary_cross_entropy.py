"""NumPy-native binary cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate, _storage
from tensors.backend.numpy.kernels.nn.losses import _reduce_losses
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
    """Compute binary cross-entropy from NumPy-native values."""
    if prediction_shape != target_shape:
        return None
    values = (
        numpy.asarray(prediction_values).reshape(prediction_shape).astype(numpy.float64)
    )
    targets = numpy.asarray(target_values).reshape(target_shape).astype(numpy.float64)
    if from_logits:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            correction = numpy.log1p(numpy.exp(-numpy.abs(values)))
            ordinary = numpy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
            losses = numpy.where(
                numpy.isposinf(values),
                numpy.where(targets == 1.0, 0.0, numpy.inf),
                numpy.where(
                    numpy.isneginf(values),
                    numpy.where(targets == 0.0, 0.0, numpy.inf),
                    ordinary,
                ),
            )
    else:
        invalid = ~numpy.isfinite(values) | (values < 0.0) | (values > 1.0)
        if bool(numpy.any(invalid)):
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(divide="ignore", invalid="ignore"):
            losses = numpy.where(
                values == 0.0,
                numpy.where(targets == 0.0, 0.0, numpy.inf),
                numpy.where(
                    values == 1.0,
                    numpy.where(targets == 1.0, 0.0, numpy.inf),
                    -targets * numpy.log(values)
                    - (1.0 - targets) * numpy.log1p(-values),
                ),
            )
    return _storage(
        _reduce_losses(losses, reduction),
        dtype=dtype,
        output_shape=output_shape,
    )
