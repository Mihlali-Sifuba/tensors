"""NumPy implementation of the binary cross-entropy VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor
    from tensors.backend.types import LossReduction


def binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested binary cross-entropy VJPs."""
    need_prediction, need_target = needs_input_grad
    values = _view(prediction).astype(numpy.float64, copy=False)
    targets = _view(target).astype(numpy.float64, copy=False)
    upstream = _view(grad).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if reduction == "none":
        if upstream.shape != values.shape:
            return None
        expanded_upstream = upstream
    else:
        scale = 1.0 / values.size if reduction == "mean" and values.size else 1.0
        try:
            expanded_upstream = (
                numpy.broadcast_to(upstream.reshape(()), values.shape) * scale
            )
        except ValueError:
            return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if from_logits:
            if not _finite_operands(values, targets, upstream):
                return None
            magnitude = numpy.exp(-numpy.abs(values))
            sigmoid = numpy.where(
                values >= 0.0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude)
            )
            prediction_derivative = sigmoid - targets
            target_derivative = -values
        else:
            invalid = ~numpy.isfinite(values) | ((values < 0.0) | (values > 1.0))
            valid = (
                numpy.all(numpy.isfinite(targets))
                & numpy.all(numpy.isfinite(upstream))
                & ~numpy.any(invalid)
            )
            if not bool(valid):
                return None
            prediction_derivative = numpy.where(
                values == 0.0,
                numpy.where(targets == 0.0, 1.0, -numpy.inf),
                numpy.where(
                    values == 1.0,
                    numpy.where(targets == 1.0, -1.0, numpy.inf),
                    (values - targets) / (values * (1.0 - values)),
                ),
            )
            target_derivative = numpy.where(
                values == 0.0,
                numpy.inf,
                numpy.where(
                    values == 1.0, -numpy.inf, numpy.log1p(-values) - numpy.log(values)
                ),
            )
        prediction_result = expanded_upstream * prediction_derivative
        target_result = expanded_upstream * target_derivative
        zero_upstream = expanded_upstream == 0.0
        prediction_result = numpy.where(zero_upstream, 0.0, prediction_result)
        target_result = numpy.where(zero_upstream, 0.0, target_result)
    prediction_storage = None
    if need_prediction:
        prediction_storage = _storage(
            prediction_result, dtype=grad.dtype, output_shape=prediction.shape
        )
        if prediction_storage is None:
            return None
    target_storage = None
    if need_target:
        target_storage = _storage(
            target_result, dtype=grad.dtype, output_shape=target.shape
        )
        if target_storage is None:
            return None
    return (prediction_storage, target_storage)
