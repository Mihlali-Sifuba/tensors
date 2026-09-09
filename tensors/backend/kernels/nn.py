"""Neural-network, probability, and loss kernels."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..storage import Storage
from .core import _errstate, _finite_operands, _numpy, _shape_size, _storage, _view
from .reductions import _normalization_terms, reduction

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor
    from ..types import (
        LossReduction,
        NormalizationOperation,
    )

def normalization(
    operation: NormalizationOperation,
    value: Tensor,
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run fused softmax or log-softmax on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    if operation == "softmax":
        result = probabilities
    else:
        with _errstate(numpy, over="ignore", invalid="ignore"):
            result = values - maximum - correction
    if not _finite_operands(values, probabilities, result, numpy=numpy):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def normalization_gradient(
    operation: NormalizationOperation,
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Storage | None:
    """Run a softmax-family VJP away from dominant cancellation."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    if upstream.shape != values.shape:
        return None
    _, _, probabilities = _normalization_terms(values, axis, numpy)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        if operation == "softmax":
            spread = numpy.max(upstream, axis=axis, keepdims=True) - numpy.min(
                upstream,
                axis=axis,
                keepdims=True,
            )
            expectation = numpy.sum(
                upstream * probabilities,
                axis=axis,
                keepdims=True,
            )
            result = probabilities * (upstream - expectation)
            result = numpy.where(spread == 0.0, 0.0, result)
        else:
            total = numpy.sum(upstream, axis=axis, keepdims=True)
            result = upstream - probabilities * total
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
        & numpy.all(numpy.isfinite(result))
    )
    if not bool(valid):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def _reduce_losses(values: Any, reduction: LossReduction, numpy: Any) -> Any:
    """Reduce non-negative losses without overflowing an ordinary mean."""
    if reduction == "none":
        return values
    if reduction == "sum":
        with _errstate(numpy, over="ignore", invalid="ignore"):
            return numpy.asarray([numpy.sum(values)])
    scale = numpy.max(values)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        exceptional = (scale == 0.0) | ~numpy.isfinite(scale)
        safe_scale = numpy.where(exceptional, 1.0, scale)
        stable = safe_scale * numpy.mean(values / safe_scale)
        result = numpy.where(exceptional, numpy.mean(values), stable)
    return numpy.asarray(result).reshape(1)

def distributions_valid(targets: Tensor, axis: int) -> bool:
    """Return whether dense targets are finite normalized probabilities."""
    numpy = _numpy()
    values = _view(targets, numpy).astype(numpy.float64, copy=False)
    valid_values = numpy.all(
        numpy.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    )
    totals = numpy.sum(values, axis=axis)
    class_count = targets.shape[axis]
    epsilon = numpy.finfo(numpy.float64).eps
    accumulated_error = (
        class_count
        * epsilon
        * numpy.sum(numpy.abs(values), axis=axis)
    )
    tolerance = numpy.maximum(
        1e-7,
        1e-7 * numpy.maximum(numpy.abs(totals), 1.0),
    )
    valid_totals = numpy.all(
        numpy.abs(totals - 1.0) + accumulated_error <= tolerance
    )
    return bool(valid_values & valid_totals)

def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused dense multiclass cross-entropy."""
    numpy = _numpy()
    values = _view(logits, numpy).astype(numpy.float64, copy=False)
    weights = _view(targets, numpy).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        log_probabilities = values - maximum - correction
        contributions = numpy.where(
            weights == 0.0,
            0.0,
            -weights * log_probabilities,
        )
        losses = numpy.sum(contributions, axis=axis)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.isnan(losses))
    )
    if not bool(valid):
        return None
    result = _reduce_losses(losses, reduction, numpy)
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested dense multiclass cross-entropy VJPs."""
    numpy = _numpy()
    need_logits, need_targets = needs_input_grad
    values = _view(logits, numpy).astype(numpy.float64, copy=False)
    weights = _view(targets, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != weights.shape:
        return None
    maximum, correction, probabilities = _normalization_terms(
        values,
        axis,
        numpy,
    )
    sample_shape = values.shape[:axis] + values.shape[axis + 1:]
    try:
        if reduction == "none":
            expanded_upstream = upstream.reshape(sample_shape)
        else:
            scale = 1.0 / _shape_size(sample_shape) if (
                reduction == "mean" and _shape_size(sample_shape)
            ) else 1.0
            expanded_upstream = numpy.broadcast_to(
                upstream.reshape(()),
                sample_shape,
            ) * scale
        expanded_upstream = numpy.expand_dims(expanded_upstream, axis=axis)
    except (TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        target_mass = numpy.sum(weights, axis=axis, keepdims=True)
        log_probabilities = values - maximum - correction
        logits_result = expanded_upstream * (
            target_mass * probabilities - weights
        )
        targets_result = -expanded_upstream * log_probabilities
        zero_upstream = expanded_upstream == 0.0
        logits_result = numpy.where(zero_upstream, 0.0, logits_result)
        targets_result = numpy.where(zero_upstream, 0.0, targets_result)
    valid = (
        numpy.all(numpy.isfinite(values))
        & numpy.all(numpy.isfinite(weights))
        & numpy.all(numpy.isfinite(upstream))
        & numpy.all(numpy.isfinite(probabilities))
        & ~numpy.any(numpy.max(probabilities, axis=axis) > 0.95)
    )
    if not bool(valid):
        return None
    logits_storage = None
    if need_logits:
        logits_storage = _storage(
            logits_result,
            dtype=grad.dtype,
            output_shape=logits.shape,
            numpy=numpy,
        )
        if logits_storage is None:
            return None
    targets_storage = None
    if need_targets:
        targets_storage = _storage(
            targets_result,
            dtype=grad.dtype,
            output_shape=targets.shape,
            numpy=numpy,
        )
        if targets_storage is None:
            return None
    return logits_storage, targets_storage

def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run fused binary cross-entropy."""
    numpy = _numpy()
    values = _view(prediction, numpy).astype(numpy.float64, copy=False)
    targets = _view(target, numpy).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if from_logits:
        if not _finite_operands(values, targets, numpy=numpy):
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            correction = numpy.log1p(numpy.exp(-numpy.abs(values)))
            losses = numpy.where(
                values >= 0.0,
                (1.0 - targets) * values + correction,
                -targets * values + correction,
            )
    else:
        invalid = (~numpy.isfinite(values)) | (
            (values < 0.0) | (values > 1.0)
        )
        target_valid = numpy.all(numpy.isfinite(targets))
        value_valid = ~numpy.any(invalid)
        status = int(
            numpy.asarray(~target_valid, dtype=numpy.uint8)
            | (
                numpy.asarray(~value_valid, dtype=numpy.uint8)
                * numpy.uint8(2)
            )
        )
        if status & 1:
            return None
        if status & 2:
            raise ValueError(
                "binary cross-entropy probabilities must be between 0 and 1"
            )
        with _errstate(numpy, divide="ignore", invalid="ignore"):
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
    result = _reduce_losses(losses, reduction, numpy)
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

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
    numpy = _numpy()
    need_prediction, need_target = needs_input_grad
    values = _view(prediction, numpy).astype(numpy.float64, copy=False)
    targets = _view(target, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != targets.shape:
        return None
    if reduction == "none":
        if upstream.shape != values.shape:
            return None
        expanded_upstream = upstream
    else:
        scale = 1.0 / values.size if reduction == "mean" and values.size else 1.0
        try:
            expanded_upstream = numpy.broadcast_to(
                upstream.reshape(()),
                values.shape,
            ) * scale
        except ValueError:
            return None

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if from_logits:
            if not _finite_operands(
                values,
                targets,
                upstream,
                numpy=numpy,
            ):
                return None
            magnitude = numpy.exp(-numpy.abs(values))
            sigmoid = numpy.where(
                values >= 0.0,
                1.0 / (1.0 + magnitude),
                magnitude / (1.0 + magnitude),
            )
            prediction_derivative = sigmoid - targets
            target_derivative = -values
        else:
            invalid = (~numpy.isfinite(values)) | (
                (values < 0.0) | (values > 1.0)
            )
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
                    values == 1.0,
                    -numpy.inf,
                    numpy.log1p(-values) - numpy.log(values),
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
            prediction_result,
            dtype=grad.dtype,
            output_shape=prediction.shape,
            numpy=numpy,
        )
        if prediction_storage is None:
            return None
    target_storage = None
    if need_target:
        target_storage = _storage(
            target_result,
            dtype=grad.dtype,
            output_shape=target.shape,
            numpy=numpy,
        )
        if target_storage is None:
            return None
    return prediction_storage, target_storage
