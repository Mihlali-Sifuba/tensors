"""Shared NumPy/CuPy kernels backed by native array storage."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, cast

from ..storage import CudaStorage, NumPyStorage, Storage

if TYPE_CHECKING:
    from .._typing import Scalar, TensorIndex
    from ..dtype import DataType
    from ..tensor import Tensor
    from . import (
        ArgExtremumOperation,
        BinaryOperation,
        ComparisonOperation,
        DifferentiableReductionOperation,
        ExtremumOperation,
        LossReduction,
        NormalizationOperation,
        ReductionOperation,
        UnaryOperation,
    )


def _view(tensor: Tensor, numpy: Any) -> Any:
    """Return a backend-native view of a tensor without repeated transfers."""
    from . import get_backend

    storage = tensor._storage_for(get_backend())
    return storage.buffer.reshape(tensor.shape)


def _cuda_integer(dtype: DataType) -> bool:
    """Return whether exact integer semantics require the Python fallback."""
    from . import get_backend

    return get_backend() == "cuda" and dtype.kind == "integer"


def _errstate(numpy: Any, **settings: str) -> Any:
    """Use NumPy warning controls when the selected array module provides them."""
    factory = getattr(numpy, "errstate", None)
    if factory is None:
        return nullcontext()
    return factory(**settings)


def _operand(value: Tensor | Scalar, dtype: DataType, numpy: Any) -> Any:
    """Return an array operand with Python-reference working precision."""
    from ..tensor import Tensor
    from . import get_backend

    if get_backend() == "cuda" and dtype.kind == "integer":
        # CuPy has no object dtype with Python's unbounded intermediate integer
        # semantics. Let the reference backend handle these operations.
        raise TypeError("CUDA integer kernels require the Python fallback")

    if isinstance(value, Tensor):
        result = _view(value, numpy)
    else:
        result = value
    working_dtype = numpy.float64 if dtype.kind == "floating" else object
    return numpy.asarray(result, dtype=working_dtype)


def _storage(
    result: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    numpy: Any,
) -> Storage | None:
    """Retain a native array result without changing tensor semantics."""
    from . import get_backend

    backend = get_backend()
    flattened = numpy.asarray(result).reshape(-1)
    if dtype.kind == "integer":
        if backend == "cuda":
            return None
        try:
            contiguous = numpy.asarray(flattened, dtype=numpy.dtype(dtype.name))
        except (OverflowError, TypeError, ValueError):
            return None
    else:
        try:
            flattened = numpy.asarray(flattened, dtype=numpy.float64)
        except (OverflowError, TypeError, ValueError):
            return None
        target_dtype = numpy.dtype(dtype.name)
        finite = numpy.isfinite(flattened)
        outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
        if bool(numpy.any(finite & outside_range)):
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            contiguous = numpy.asarray(flattened, dtype=target_dtype)
    storage: Storage
    if backend == "cuda":
        storage = CudaStorage(contiguous, dtype)
    else:
        storage = NumPyStorage(contiguous, dtype)
    if storage.size != _shape_size(output_shape):
        raise RuntimeError("Array kernel returned an unexpected result size")
    return storage


def binary(
    operation: BinaryOperation,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting NumPy binary kernel."""
    numpy = _numpy()
    try:
        left_array = _operand(left, dtype, numpy)
        right_array = _operand(right, dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    functions = {
        "add": numpy.add,
        "subtract": numpy.subtract,
        "multiply": numpy.multiply,
        "divide": numpy.true_divide,
        "power": numpy.power,
    }
    if operation == "divide" and bool(numpy.any(right_array == 0)):
        raise ZeroDivisionError("Division by zero")
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        result = functions[operation](left_array, right_array)

    if operation == "power" and dtype.kind == "floating" and _finite_operands(
        left_array,
        right_array,
        numpy=numpy,
    ) and not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run elementwise NumPy negation."""
    numpy = _numpy()
    try:
        operand = _operand(value, dtype, numpy)
    except (TypeError, ValueError):
        return None
    result = numpy.negative(operand)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def unary(
    operation: UnaryOperation,
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run an elementwise unary kernel while preserving public domains."""
    if dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        values = _view(value, numpy).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None

    if operation == "sqrt" and bool(numpy.any(values < 0.0)):
        raise ValueError("sqrt is only defined for non-negative values")
    if operation == "log" and bool(numpy.any(values <= 0.0)):
        raise ValueError("log is only defined for positive values")
    if operation in {"arcsin", "arccos"} and bool(
        numpy.any((values < -1.0) | (values > 1.0))
    ):
        raise ValueError(
            f"{operation} is only defined for values between -1 and 1"
        )
    if operation == "arccosh" and bool(numpy.any(values < 1.0)):
        raise ValueError(
            "arccosh is only defined for values greater than or equal to 1"
        )
    if operation == "arctanh":
        outside = (~numpy.isnan(values)) & (
            (values <= -1.0) | (values >= 1.0)
        )
        if bool(numpy.any(outside)):
            raise ValueError(
                "arctanh is only defined for values strictly between -1 and 1"
            )
    if operation in {"sin", "cos", "tan"} and bool(
        numpy.any(numpy.isinf(values))
    ):
        return None

    functions = {
        "abs": numpy.abs,
        "sqrt": numpy.sqrt,
        "exp": numpy.exp,
        "log": numpy.log,
        "sin": numpy.sin,
        "cos": numpy.cos,
        "tan": numpy.tan,
        "arcsin": numpy.arcsin,
        "arccos": numpy.arccos,
        "arctan": numpy.arctan,
        "sinh": numpy.sinh,
        "cosh": numpy.cosh,
        "arcsinh": numpy.arcsinh,
        "arccosh": numpy.arccosh,
        "arctanh": numpy.arctanh,
        "sign": numpy.sign,
        "tanh": numpy.tanh,
    }
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "relu":
            result = numpy.where(numpy.isnan(values), values, numpy.maximum(values, 0.0))
        elif operation == "sigmoid":
            magnitude = numpy.exp(-numpy.abs(values))
            result = numpy.where(
                values >= 0.0,
                1.0 / (1.0 + magnitude),
                magnitude / (1.0 + magnitude),
            )
        elif operation == "softplus":
            result = numpy.log1p(numpy.exp(-numpy.abs(values))) + numpy.maximum(
                values,
                0.0,
            )
        else:
            result = functions[operation](values)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def unary_gradient(
    operation: UnaryOperation,
    grad: Tensor,
    value: Tensor,
) -> Storage | None:
    """Run the vector-Jacobian product for an elementwise unary operation."""
    numpy = _numpy()
    try:
        upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
        values = _view(value, numpy).astype(numpy.float64, copy=False)
    except (TypeError, ValueError):
        return None
    if upstream.shape != values.shape:
        return None

    if operation == "sqrt":
        if bool(numpy.any(values == 0.0)):
            raise ValueError("sqrt derivative is undefined at zero")
        if bool(numpy.any(values < 0.0)):
            return None
    if operation in {"arcsin", "arccos"}:
        if bool(numpy.any((values == -1.0) | (values == 1.0))):
            raise ValueError(
                f"{operation} derivative is undefined at -1 and 1"
            )
        if bool(numpy.any((values < -1.0) | (values > 1.0))):
            return None
    if operation == "arccosh":
        if bool(numpy.any(values == 1.0)):
            raise ValueError("arccosh derivative is undefined at 1")
        if bool(numpy.any(values < 1.0)):
            return None
    if operation == "sign" and bool(numpy.any(values == 0.0)):
        raise ValueError("sign derivative is undefined at zero")
    if operation in {"sin", "cos", "tan"} and bool(
        numpy.any(numpy.isinf(values))
    ):
        return None

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "abs":
            derivative = numpy.where(
                numpy.isnan(values),
                numpy.nan,
                numpy.where(values > 0.0, 1.0, numpy.where(values < 0.0, -1.0, 0.0)),
            )
        elif operation == "sqrt":
            derivative = 1.0 / (2.0 * numpy.sqrt(values))
        elif operation == "exp":
            derivative = numpy.exp(values)
        elif operation == "log":
            derivative = 1.0 / values
        elif operation == "sin":
            derivative = numpy.cos(values)
        elif operation == "cos":
            derivative = -numpy.sin(values)
        elif operation == "tan":
            cosine = numpy.cos(values)
            if bool(numpy.any(numpy.abs(cosine) < numpy.finfo(numpy.float64).eps)):
                return None
            derivative = 1.0 / (cosine * cosine)
        elif operation == "arcsin":
            derivative = 1.0 / numpy.sqrt(1.0 - values * values)
        elif operation == "arccos":
            derivative = -1.0 / numpy.sqrt(1.0 - values * values)
        elif operation == "arctan":
            reciprocal = 1.0 / numpy.abs(values)
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                numpy.where(
                    numpy.abs(values) <= 1.0,
                    1.0 / (1.0 + values * values),
                    (reciprocal * reciprocal) / (1.0 + reciprocal * reciprocal),
                ),
            )
        elif operation == "sinh":
            derivative = numpy.cosh(values)
        elif operation == "cosh":
            derivative = numpy.sinh(values)
        elif operation == "arcsinh":
            reciprocal = 1.0 / numpy.abs(values)
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                numpy.where(
                    numpy.abs(values) <= 1.0,
                    1.0 / numpy.sqrt(1.0 + values * values),
                    reciprocal / numpy.sqrt(1.0 + reciprocal * reciprocal),
                ),
            )
        elif operation == "arccosh":
            derivative = numpy.where(
                numpy.isinf(values),
                0.0,
                1.0 / (numpy.sqrt(values - 1.0) * numpy.sqrt(values + 1.0)),
            )
        elif operation == "arctanh":
            derivative = 1.0 / (1.0 - values * values)
        elif operation == "sign":
            derivative = numpy.where(numpy.isnan(values), numpy.nan, 0.0)
        elif operation == "relu":
            derivative = numpy.where(
                numpy.isnan(values),
                numpy.nan,
                numpy.where(values > 0.0, 1.0, 0.0),
            )
        elif operation in {"sigmoid", "softplus"}:
            magnitude = numpy.exp(-numpy.abs(values))
            if operation == "sigmoid":
                derivative = magnitude / ((1.0 + magnitude) ** 2.0)
            else:
                derivative = numpy.where(
                    values >= 0.0,
                    1.0 / (1.0 + magnitude),
                    magnitude / (1.0 + magnitude),
                )
        else:
            magnitude = numpy.exp(-2.0 * numpy.abs(values))
            derivative = 4.0 * magnitude / ((1.0 + magnitude) ** 2.0)
        result = upstream * derivative
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def _normalization_terms(
    values: Any,
    axis: int | tuple[int, ...],
    numpy: Any,
) -> tuple[Any, Any, Any] | None:
    """Return stable maxima, corrections, and probabilities."""
    if not bool(numpy.all(numpy.isfinite(values))):
        return None
    maximum = numpy.max(values, axis=axis, keepdims=True)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        deltas = values - maximum
        maxima = numpy.sum(deltas == 0.0, axis=axis, keepdims=True)
        tails = numpy.sum(
            numpy.where(deltas == 0.0, 0.0, numpy.exp(deltas)),
            axis=axis,
            keepdims=True,
        )
        correction = numpy.log(maxima) + numpy.log1p(tails / maxima)
        probabilities = numpy.exp(deltas - correction)
    if not bool(numpy.all(numpy.isfinite(probabilities))):
        return None
    return maximum, correction, probabilities


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
    terms = _normalization_terms(values, axis, numpy)
    if terms is None:
        return None
    maximum, correction, probabilities = terms
    if operation == "softmax":
        result = probabilities
    else:
        with _errstate(numpy, over="ignore", invalid="ignore"):
            result = values - maximum - correction
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
    if upstream.shape != values.shape or not bool(
        numpy.all(numpy.isfinite(upstream))
    ):
        return None
    terms = _normalization_terms(values, axis, numpy)
    if terms is None:
        return None
    probabilities = terms[2]
    if bool(numpy.any(numpy.max(probabilities, axis=axis) > 0.95)):
        return None
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
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a stable log-sum-exp reduction on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    terms = _normalization_terms(values, axes, numpy)
    if terms is None:
        return None
    maximum, correction, _ = terms
    with _errstate(numpy, over="ignore", invalid="ignore"):
        result = maximum + correction
    if not keepdims and axes:
        result = numpy.squeeze(result, axis=axes)
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def logsumexp_gradient(
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run a stable log-sum-exp VJP on finite values."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if not bool(numpy.all(numpy.isfinite(upstream))):
        return None
    terms = _normalization_terms(values, axes, numpy)
    if terms is None:
        return None
    probabilities = terms[2]
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = expanded * probabilities
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
    if scale == 0.0 or not numpy.isfinite(scale):
        return numpy.asarray([numpy.mean(values)])
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        return numpy.asarray([scale * numpy.mean(values / scale)])


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
    if values.shape != weights.shape or not bool(
        numpy.all(numpy.isfinite(weights))
    ):
        return None
    terms = _normalization_terms(values, axis, numpy)
    if terms is None:
        return None
    maximum, correction, _ = terms
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        log_probabilities = values - maximum - correction
        contributions = numpy.where(
            weights == 0.0,
            0.0,
            -weights * log_probabilities,
        )
        losses = numpy.sum(contributions, axis=axis)
    if bool(numpy.any(numpy.isnan(losses))):
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
) -> tuple[Storage, Storage] | None:
    """Run fused dense multiclass cross-entropy VJPs."""
    numpy = _numpy()
    values = _view(logits, numpy).astype(numpy.float64, copy=False)
    weights = _view(targets, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != weights.shape or not _finite_operands(
        weights,
        upstream,
        numpy=numpy,
    ):
        return None
    terms = _normalization_terms(values, axis, numpy)
    if terms is None:
        return None
    maximum, correction, probabilities = terms
    if bool(numpy.any(numpy.max(probabilities, axis=axis) > 0.95)):
        return None
    sample_shape = values.shape[:axis] + values.shape[axis + 1:]
    try:
        if reduction == "none":
            expanded_upstream = upstream.reshape(sample_shape)
        else:
            scale = 1.0 / _shape_size(sample_shape) if (
                reduction == "mean" and _shape_size(sample_shape)
            ) else 1.0
            expanded_upstream = numpy.full(sample_shape, upstream.item() * scale)
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
    logits_storage = _storage(
        logits_result,
        dtype=grad.dtype,
        output_shape=logits.shape,
        numpy=numpy,
    )
    targets_storage = _storage(
        targets_result,
        dtype=grad.dtype,
        output_shape=targets.shape,
        numpy=numpy,
    )
    if logits_storage is None or targets_storage is None:
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
    if values.shape != targets.shape or not bool(
        numpy.all(numpy.isfinite(targets))
    ):
        return None
    if from_logits:
        if not bool(numpy.all(numpy.isfinite(values))):
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
        if bool(numpy.any(invalid)):
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
) -> tuple[Storage, Storage] | None:
    """Run fused binary cross-entropy VJPs."""
    numpy = _numpy()
    values = _view(prediction, numpy).astype(numpy.float64, copy=False)
    targets = _view(target, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    if values.shape != targets.shape or not bool(
        numpy.all(numpy.isfinite(targets))
    ):
        return None
    if reduction == "none":
        if upstream.shape != values.shape:
            return None
        expanded_upstream = upstream
    else:
        scale = 1.0 / values.size if reduction == "mean" and values.size else 1.0
        try:
            expanded_upstream = numpy.full(values.shape, upstream.item() * scale)
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
            if not bool(numpy.all(numpy.isfinite(values))):
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
            if bool(numpy.any(invalid)):
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
    prediction_storage = _storage(
        prediction_result,
        dtype=grad.dtype,
        output_shape=prediction.shape,
        numpy=numpy,
    )
    target_storage = _storage(
        target_result,
        dtype=grad.dtype,
        output_shape=target.shape,
        numpy=numpy,
    )
    if prediction_storage is None or target_storage is None:
        return None
    return prediction_storage, target_storage


def slice_tensor(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a NumPy slicing kernel after caller-side key validation."""
    numpy = _numpy()
    try:
        result = _view(value, numpy)[key]
    except ValueError:
        return None
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Scatter flat values into a zero NumPy tensor."""
    if _cuda_integer(value.dtype):
        return None
    numpy = _numpy()
    working_dtype = object if value.dtype.kind == "integer" else numpy.dtype(
        value.dtype.name
    )
    result = numpy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        values = _view(value, numpy).reshape(-1).astype(
            working_dtype,
            copy=False,
        )
    except ValueError:
        return None
    numpy.add.at(result, indices, values)
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    try:
        source = _view(value, numpy).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = numpy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        result = source.astype(numpy.float64, copy=False)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create constant-filled canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.full(shape, fill_value, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=shape,
        numpy=numpy,
    )


def eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create identity-like canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.eye(rows, columns, k=k, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(rows, columns),
        numpy=numpy,
    )


def arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create an arithmetic progression from a validated element count."""
    integer_inputs = isinstance(start, int) and isinstance(step, int)
    if _cuda_integer(dtype) or (_is_cuda() and integer_inputs):
        return None
    numpy = _numpy()
    working_dtype = object if integer_inputs else numpy.float64
    indices = numpy.arange(count, dtype=working_dtype)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start + indices * step
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )


def _is_cuda() -> bool:
    from . import get_backend

    return get_backend() == "cuda"


def linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create evenly spaced values when ordinary vector arithmetic is safe."""
    if count < 2:
        return None
    start_value = float(start)
    stop_value = float(stop)
    if start_value * stop_value < 0.0:
        return None
    numpy = _numpy()
    limit = numpy.finfo(numpy.float64).max / 4.0
    if abs(start_value) > limit or abs(stop_value) > limit:
        return None
    fractions = numpy.arange(count, dtype=numpy.float64) / (count - 1)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start_value * (1.0 - fractions) + stop_value * fractions
    result[0] = start
    result[-1] = stop
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )


def transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Permute tensor axes into canonical contiguous storage."""
    numpy = _numpy()
    result = numpy.transpose(_view(value, numpy), axes=permutation)
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Concatenate tensors along an existing axis."""
    numpy = _numpy()
    try:
        result = numpy.concatenate(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Stack tensors along a newly inserted axis."""
    numpy = _numpy()
    try:
        result = numpy.stack(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a vector outer product."""
    numpy = _numpy()
    try:
        left_values = _operand(left, dtype, numpy)
        right_values = _operand(right, dtype, numpy)
    except (TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.multiply.outer(left_values, right_values)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(left.size, right.size),
        numpy=numpy,
    )


def outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Run stable native outer-product VJPs when products can be summed."""
    numpy = _numpy()
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left_values = _view(left, numpy).astype(numpy.float64, copy=False)
    right_values = _view(right, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(
        upstream,
        left_values,
        right_values,
        numpy=numpy,
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        left_terms = upstream * right_values
        right_terms = upstream * left_values[:, None]
    if not _stable_sum_candidate(left_terms, (1,), numpy) or not (
        _stable_sum_candidate(right_terms, (0,), numpy)
    ):
        return None
    left_result = numpy.sum(left_terms, axis=1)
    right_result = numpy.sum(right_terms, axis=0)
    left_storage = _storage(
        left_result,
        dtype=grad.dtype,
        output_shape=left.shape,
        numpy=numpy,
    )
    right_storage = _storage(
        right_result,
        dtype=grad.dtype,
        output_shape=right.shape,
        numpy=numpy,
    )
    if left_storage is None or right_storage is None:
        return None
    return left_storage, right_storage


def sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> Storage | None:
    """Apply one fused SGD update."""
    numpy = _numpy()
    values = _view(parameter, numpy).astype(numpy.float64, copy=False)
    gradients = _view(gradient, numpy).astype(numpy.float64, copy=False)
    if not _finite_operands(values, gradients, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = values - learning_rate * gradients
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=parameter.dtype,
        output_shape=parameter.shape,
        numpy=numpy,
    )


def adam_update(
    parameter: Tensor,
    gradient: Tensor,
    moment: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_correction: float,
    second_correction: float,
) -> tuple[Storage, Storage, Storage, Storage, Storage] | None:
    """Apply one fused Adam update on finite, non-cancelling state."""
    numpy = _numpy()
    tensors = (parameter, gradient, moment, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, moments, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    left_term = beta1 * moments
    right_term = (1.0 - beta1) * gradients
    if bool(numpy.any((left_term > 0.0) & (right_term < 0.0))) or bool(
        numpy.any((left_term < 0.0) & (right_term > 0.0))
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_moments = left_term + right_term
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            beta2 * scaled_values * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_correction = numpy.sqrt(second_correction)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        denominator = first_correction * (
            root_moment + epsilon * root_correction
        )
        ratio = new_moments * root_correction / denominator
        parameter_result = parameter_values - learning_rate * ratio
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(
        new_moments,
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (new_moments, gradient.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        _storage(
            result,
            dtype=dtype,
            output_shape=gradient.shape,
            numpy=numpy,
        )
        for result, dtype in specifications
    )
    if any(storage is None for storage in storages):
        return None
    return cast(
        "tuple[Storage, Storage, Storage, Storage, Storage]",
        storages,
    )


def rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage, Storage] | None:
    """Apply one fused RMSprop update on finite optimizer state."""
    numpy = _numpy()
    tensors = (parameter, gradient, scale, scaled)
    values = [
        _view(item, numpy).astype(numpy.float64, copy=False)
        for item in tensors
    ]
    parameter_values, gradients, scales, scaled_values = values
    if not _finite_operands(*values, numpy=numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            rho * scaled_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        parameter_result = parameter_values - (
            learning_rate * gradients / (root_moment + epsilon)
        )
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(
        new_scales,
        new_scaled,
        parameter_result,
        numpy=numpy,
    ):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        _storage(
            result,
            dtype=dtype,
            output_shape=gradient.shape,
            numpy=numpy,
        )
        for result, dtype in specifications
    )
    if any(storage is None for storage in storages):
        return None
    return cast(
        "tuple[Storage, Storage, Storage, Storage]",
        storages,
    )


def reduction(
    operation: ReductionOperation,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None

    numpy = _numpy()
    axis = axes
    if operation in {"min", "max"}:
        values = _view(value, numpy)
        function = numpy.min if operation == "min" else numpy.max
        result = function(values, axis=axis, keepdims=keepdims)
    elif operation == "prod":
        values = _view(value, numpy)
        working = (
            values.astype(object)
            if dtype.kind == "integer"
            else values.astype(numpy.float64, copy=False)
        )
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = numpy.prod(working, axis=axis, keepdims=keepdims)
    else:
        values = _view(value, numpy).astype(numpy.float64, copy=False)
        if not bool(numpy.all(numpy.isfinite(values))):
            return None

    if operation in {"sum", "mean"}:
        if operation == "sum" and dtype.kind == "integer":
            return None
        has_positive = bool(numpy.any(values > 0.0))
        has_negative = bool(numpy.any(values < 0.0))
        has_subnormal = bool(
            numpy.any(
                (values != 0.0)
                & (numpy.abs(values) < numpy.finfo(numpy.float64).tiny)
            )
        )
        if (has_positive and has_negative) or has_subnormal:
            return None
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            if operation == "sum":
                result = numpy.sum(values, axis=axis, keepdims=keepdims)
            else:
                result = numpy.mean(values, axis=axis, keepdims=keepdims)
        if not bool(numpy.all(numpy.isfinite(result))):
            return None
    elif operation in {"variance", "std"}:
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            center = numpy.mean(values, axis=axis, keepdims=True)
            centered = values - center
        if not bool(numpy.all(numpy.isfinite(centered))):
            return None
        scale = numpy.max(numpy.abs(centered), axis=axis, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized = centered / safe_scale
        normalized_variance = numpy.mean(
            normalized * normalized,
            axis=axis,
            keepdims=keepdims,
        )
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            deviation = output_scale * numpy.sqrt(normalized_variance)
            result = deviation * deviation if operation == "variance" else deviation
    elif operation == "norm":
        absolute = numpy.abs(values)
        scale = numpy.max(absolute, axis=axis, keepdims=True)
        safe_scale = numpy.where(scale == 0.0, 1.0, scale)
        normalized = values / safe_scale
        normalized_magnitude = numpy.sqrt(
            numpy.sum(
                normalized * normalized,
                axis=axis,
                keepdims=keepdims,
            )
        )
        output_scale = scale if keepdims else numpy.squeeze(scale, axis=axis)
        with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
            result = output_scale * normalized_magnitude

    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def reduction_gradient(
    operation: DifferentiableReductionOperation,
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> Storage | None:
    """Run VJPs for standard deviation, product, and extrema reductions."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    expanded_shape = tuple(
        1 if dimension in axes else size
        for dimension, size in enumerate(value.shape)
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= value.shape[axis]

    with _errstate(

        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        if operation == "std":
            if not bool(numpy.all(numpy.isfinite(values))) or count == 0:
                return None
            scale = numpy.max(numpy.abs(values), axis=axes, keepdims=True)
            safe_scale = numpy.where(scale == 0.0, 1.0, scale)
            normalized = values / safe_scale
            center = numpy.mean(normalized, axis=axes, keepdims=True)
            centered = normalized - center
            deviation = numpy.sqrt(
                numpy.mean(centered * centered, axis=axes, keepdims=True)
            )
            derivative = numpy.where(
                deviation == 0.0,
                0.0,
                centered / (count * deviation),
            )
            result = expanded * derivative
        elif operation == "prod":
            if not bool(numpy.all(numpy.isfinite(values))):
                return None
            zero_count = numpy.sum(values == 0.0, axis=axes, keepdims=True)
            product = numpy.prod(values, axis=axes, keepdims=True)
            nonzero_product = numpy.prod(
                numpy.where(values == 0.0, 1.0, values),
                axis=axes,
                keepdims=True,
            )
            unsafe = (
                ((zero_count == 0) & ((product == 0.0) | ~numpy.isfinite(product)))
                | (
                    (zero_count == 1)
                    & (
                        (nonzero_product == 0.0)
                        | ~numpy.isfinite(nonzero_product)
                    )
                )
            )
            if bool(numpy.any(unsafe)):
                return None
            derivative = numpy.where(
                zero_count == 0,
                product / values,
                numpy.where(
                    (zero_count == 1) & (values == 0.0),
                    nonzero_product,
                    0.0,
                ),
            )
            result = expanded * derivative
        else:
            has_nan = numpy.any(numpy.isnan(values), axis=axes, keepdims=True)
            function = numpy.min if operation == "min" else numpy.max
            extreme = function(values, axis=axes, keepdims=True)
            selected = values == extreme
            ties = numpy.sum(selected, axis=axes, keepdims=True)
            result = numpy.where(
                has_nan,
                numpy.nan,
                expanded * selected / ties,
            )
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def arg_extremum(
    operation: ArgExtremumOperation,
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a first-occurrence argmin or argmax reduction."""
    if value.size == 0:
        return None
    from ..dtype import int64

    numpy = _numpy()
    values = _view(value, numpy)
    function = numpy.argmin if operation == "argmin" else numpy.argmax
    result = function(values, axis=axis, keepdims=keepdims)
    return _storage(
        result,
        dtype=int64,
        output_shape=output_shape,
        numpy=numpy,
    )


def comparison(
    operation: ComparisonOperation,
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise comparison."""
    from ..dtype import uint8

    numpy = _numpy()
    functions = {
        "equal": numpy.equal,
        "not_equal": numpy.not_equal,
        "less": numpy.less,
        "less_equal": numpy.less_equal,
        "greater": numpy.greater,
        "greater_equal": numpy.greater_equal,
    }
    try:
        result = functions[operation](_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=uint8,
        output_shape=output_shape,
        numpy=numpy,
    )


def where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run broadcasting elementwise selection."""
    numpy = _numpy()
    try:
        result = numpy.where(
            _view(condition, numpy) != 0,
            _view(left, numpy),
            _view(right, numpy),
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def where_gradient(
    grad: Tensor,
    condition: Tensor,
) -> tuple[Storage, Storage] | None:
    """Split a selection VJP into its expanded left and right terms."""
    numpy = _numpy()
    try:
        selected = numpy.broadcast_to(_view(condition, numpy), grad.shape) != 0
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    left = _storage(
        numpy.where(selected, upstream, 0.0),
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    right = _storage(
        numpy.where(selected, 0.0, upstream),
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    if left is None or right is None:
        return None
    return left, right


def clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Clip tensor values to optional scalar bounds."""
    numpy = _numpy()
    values = _view(value, numpy)
    result = numpy.clip(values, min_value, max_value)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    numpy = _numpy()
    values = _view(value, numpy).astype(numpy.float64, copy=False)
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    mask = numpy.ones(value.shape, dtype=bool)
    if min_value is not None:
        mask &= values > min_value
    if max_value is not None:
        mask &= values < max_value
    result = numpy.where(numpy.isnan(values), numpy.nan, numpy.where(mask, upstream, 0.0))
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=value.shape,
        numpy=numpy,
    )


def extremum(
    operation: ExtremumOperation,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a broadcasting elementwise minimum or maximum."""
    numpy = _numpy()
    function = numpy.minimum if operation == "minimum" else numpy.maximum
    try:
        result = function(_view(left, numpy), _view(right, numpy))
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def extremum_gradient(
    operation: ExtremumOperation,
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Split an elementwise-extremum VJP, sharing exact ties."""
    numpy = _numpy()
    try:
        left_values, right_values = numpy.broadcast_arrays(
            _view(left, numpy),
            _view(right, numpy),
        )
    except ValueError:
        return None
    upstream = _view(grad, numpy).astype(numpy.float64, copy=False)
    has_nan = numpy.isnan(left_values) | numpy.isnan(right_values)
    ties = left_values == right_values
    left_selected = (
        left_values > right_values
        if operation == "maximum"
        else left_values < right_values
    )
    left_weight = numpy.where(
        has_nan,
        numpy.nan,
        numpy.where(ties, 0.5, numpy.where(left_selected, 1.0, 0.0)),
    )
    right_weight = numpy.where(
        has_nan,
        numpy.nan,
        numpy.where(ties, 0.5, numpy.where(left_selected, 0.0, 1.0)),
    )
    left_storage = _storage(
        upstream * left_weight,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    right_storage = _storage(
        upstream * right_weight,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )
    if left_storage is None or right_storage is None:
        return None
    return left_storage, right_storage


def _sum_axes(
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return padded target shape and axes reduced after broadcasting."""
    if len(target_shape) > len(source_shape):
        return None
    padded = (1,) * (len(source_shape) - len(target_shape)) + target_shape
    axes = []
    for axis, (source, target) in enumerate(zip(source_shape, padded)):
        if source == target:
            continue
        if target != 1:
            return None
        axes.append(axis)
    return padded, tuple(axes)


def _stable_sum_candidate(values: Any, axes: tuple[int, ...], numpy: Any) -> bool:
    """Return whether ordinary NumPy summation preserves guarded semantics."""
    if not bool(numpy.all(numpy.isfinite(values))):
        return False
    absolute = numpy.abs(values)
    if bool(
        numpy.any(
            (values != 0.0)
            & (absolute < numpy.finfo(numpy.float64).tiny)
        )
    ):
        return False
    if not axes:
        return True
    positive = numpy.any(values > 0.0, axis=axes, keepdims=True)
    negative = numpy.any(values < 0.0, axis=axes, keepdims=True)
    return not bool(numpy.any(positive & negative))


def sum_to_shape(
    gradient: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    layout = _sum_axes(gradient.shape, shape)
    if layout is None:
        return None
    _, axes = layout
    numpy = _numpy()
    values = _view(gradient, numpy).astype(numpy.float64, copy=False)
    if not _stable_sum_candidate(values, axes, numpy):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.sum(values, axis=axes, keepdims=True) if axes else values
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )


def sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Multiply and reduce broadcast VJP terms in one guarded kernel."""
    if gradient.dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        left, right = numpy.broadcast_arrays(
            _view(gradient, numpy).astype(numpy.float64, copy=False),
            _view(factor, numpy).astype(numpy.float64, copy=False),
        )
    except ValueError:
        return None
    layout = _sum_axes(tuple(left.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    if not bool(numpy.all(numpy.isfinite(left))) or not bool(
        numpy.all(numpy.isfinite(right))
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        products = left * right
    lost_range = (~numpy.isfinite(products)) | (
        (products == 0.0) & (left != 0.0) & (right != 0.0)
    )
    if bool(numpy.any(lost_range)) or not _stable_sum_candidate(
        products,
        axes,
        numpy,
    ):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = (
            numpy.sum(products, axis=axes, keepdims=True)
            if axes
            else products
        )
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )


def division_denominator_gradient(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
) -> Storage | None:
    """Calculate ``-grad * numerator / denominator**2`` when range-safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        values = _operand(numerator, grad.dtype, numpy)
        divisors = _operand(denominator, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        squares = numpy.square(divisors)
    if _finite_operands(
        upstream,
        values,
        divisors,
        numpy=numpy,
    ) and bool(numpy.any((squares == 0.0) | ~numpy.isfinite(squares))):
        return None
    with _errstate(
        numpy,
        divide="ignore",
        over="ignore",
        under="ignore",
        invalid="ignore",
    ):
        result = -upstream * values / squares
    if _unsafe_finite_result(
        result,
        upstream,
        values,
        divisors,
        numpy=numpy,
    ):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def power_base_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Calculate the power gradient with respect to its base when safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        bases = _operand(base, grad.dtype, numpy)
        powers = _operand(exponent, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    if not _finite_operands(upstream, bases, powers, numpy=numpy):
        return None
    if bool(numpy.any(bases <= 0.0)):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        power_term = numpy.power(bases, powers - 1.0)
    if bool(numpy.any((power_term == 0.0) | ~numpy.isfinite(power_term))):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = upstream * powers * power_term
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def power_exponent_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Calculate the power gradient with respect to its exponent when safe."""
    numpy = _numpy()
    try:
        upstream = _operand(grad, grad.dtype, numpy)
        bases = _operand(base, grad.dtype, numpy)
        powers = _operand(exponent, grad.dtype, numpy)
    except (OverflowError, TypeError, ValueError):
        return None
    if not _finite_operands(upstream, bases, powers, numpy=numpy):
        return None
    if bool(numpy.any(bases <= 0.0)):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        outputs = numpy.power(bases, powers)
    if bool(numpy.any((outputs == 0.0) | ~numpy.isfinite(outputs))):
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = upstream * outputs * numpy.log(bases)
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=grad.dtype,
        output_shape=grad.shape,
        numpy=numpy,
    )


def _finite_operands(*operands: Any, numpy: Any) -> bool:
    return all(
        bool(numpy.all(numpy.isfinite(operand)))
        for operand in operands
    )


def _unsafe_finite_result(
    result: Any,
    *operands: Any,
    numpy: Any,
) -> bool:
    return _finite_operands(*operands, numpy=numpy) and not bool(
        numpy.all(numpy.isfinite(result))
    )


def _numpy() -> Any:
    """Import the array module selected by the active backend."""
    from . import get_backend

    backend = get_backend()
    module_name = "cupy" if backend == "cuda" else "numpy"
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        from . import BackendUnavailableError

        raise BackendUnavailableError(
            f"The {backend} backend became unavailable after it was selected."
        ) from error


def matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return a NumPy matrix product or defer to the reference implementation."""
    if dtype.kind != "floating":
        return None

    numpy = _numpy()

    # The reference implementation accumulates float32 products in Python's
    # double precision and casts only the final result. Match that behavior by
    # using float64 as the NumPy working dtype for every supported float result.
    try:
        left_array = _view(left, numpy).astype(numpy.float64, copy=False)
        right_array = _view(right, numpy).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = numpy.matmul(left_array, right_array)

    # NumPy may overflow an intermediate product that the Python reference can
    # recover through exact-ratio summation. Preserve those semantics by asking
    # the caller to use the reference implementation whenever the fast result is
    # non-finite or cannot be represented by the requested output dtype.
    if not bool(numpy.all(numpy.isfinite(result))):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size
