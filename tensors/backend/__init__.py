"""Numerical backend selection and internal kernel dispatch."""

from __future__ import annotations

import importlib.util
import os
import threading
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from .._typing import Scalar, TensorIndex
    from ..dtype import DataType
    from ..tensor import Tensor


BackendName: TypeAlias = Literal["python", "numpy"]
BackendSelection: TypeAlias = Literal["python", "numpy", "auto"]
BinaryOperation: TypeAlias = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
]
ReductionOperation: TypeAlias = Literal[
    "sum",
    "mean",
    "variance",
    "std",
    "prod",
    "min",
    "max",
    "norm",
]
DifferentiableReductionOperation: TypeAlias = Literal[
    "std",
    "prod",
    "min",
    "max",
]
ComparisonOperation: TypeAlias = Literal[
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
]
ExtremumOperation: TypeAlias = Literal["minimum", "maximum"]
ArgExtremumOperation: TypeAlias = Literal["argmin", "argmax"]
NormalizationOperation: TypeAlias = Literal[
    "softmax",
    "log_softmax",
]
LossReduction: TypeAlias = Literal["none", "mean", "sum"]
UnaryOperation: TypeAlias = Literal[
    "abs",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "sign",
    "relu",
    "sigmoid",
    "tanh",
    "softplus",
]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected optional backend is unavailable."""


_VALID_BACKENDS = {"python", "numpy", "auto"}
_NUMPY_ELEMENTWISE_MIN_SIZE = 32
_NUMPY_REDUCTION_MIN_SIZE = 8
_NUMPY_MATMUL_MIN_WORK = 32
_backend_lock = threading.RLock()
_backend_override: ContextVar[BackendName | None] = ContextVar(
    "tensors_backend_override",
    default=None,
)


def _numpy_available() -> bool:
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> tuple[BackendName, ...]:
    """Return the numerical backends available in this environment."""
    if _numpy_available():
        return ("python", "numpy")
    return ("python",)


def _resolve_backend(backend: str) -> BackendName:
    normalized = backend.strip().lower()
    if normalized not in _VALID_BACKENDS:
        choices = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of: {choices}"
        )
    if normalized == "auto":
        return "numpy" if _numpy_available() else "python"
    if normalized == "numpy" and not _numpy_available():
        raise BackendUnavailableError(
            "The NumPy backend is unavailable. Install it with "
            "`pip install \"ms-tensors[numpy]\"`."
        )
    return cast(BackendName, normalized)


def _environment_default() -> BackendName:
    configured = os.environ.get("TENSORS_BACKEND", "python")
    return _resolve_backend(configured)


_process_backend = _environment_default()


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def _numpy_work_is_large_enough(work: int, minimum: int) -> bool:
    return get_backend() == "numpy" and work >= minimum


def get_backend() -> BackendName:
    """Return the backend active in the current execution context."""
    override = _backend_override.get()
    if override is not None:
        return override
    with _backend_lock:
        return _process_backend


def set_backend(backend: BackendSelection) -> None:
    """Set the process-wide default backend.

    Configure the default before starting worker threads. Existing scoped
    overrides created by :func:`use_backend` remain active until their contexts
    exit.
    """
    selected = _resolve_backend(backend)
    global _process_backend
    with _backend_lock:
        _process_backend = selected


@contextmanager
def use_backend(backend: BackendSelection) -> Iterator[None]:
    """Temporarily select a backend in the current execution context."""
    selected = _resolve_backend(backend)
    token = _backend_override.set(selected)
    try:
        yield
    finally:
        _backend_override.reset(token)


def execute_matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an accelerated matrix product or request the Python fallback."""
    contraction_size = left.shape[-1]
    work = _shape_size(output_shape) * contraction_size
    if not _numpy_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    from .numpy import matmul

    return matmul(left, right, dtype=dtype, output_shape=output_shape)


def execute_binary(
    operation: BinaryOperation,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an accelerated binary operation or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import binary

    return binary(
        operation,
        left,
        right,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_negate(
    value: Tensor,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run accelerated elementwise negation or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import negate

    return negate(value, dtype=dtype)


def execute_unary(
    operation: UnaryOperation,
    value: Tensor,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run an accelerated unary transform or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import unary

    return unary(operation, value, dtype=dtype)


def execute_unary_gradient(
    operation: UnaryOperation,
    grad: Tensor,
    value: Tensor,
) -> array[Any] | None:
    """Run an accelerated unary VJP or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        max(grad.size, value.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import unary_gradient

    return unary_gradient(operation, grad, value)


def execute_slice(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run accelerated tensor slicing or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import slice_tensor

    return slice_tensor(value, key, output_shape=output_shape)


def execute_slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run accelerated slice scattering or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import slice_scatter

    return slice_scatter(value, indices, output_shape=output_shape)


def execute_cast(
    value: Tensor,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run accelerated dtype conversion or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import cast_tensor

    return cast_tensor(value, dtype=dtype)


def execute_full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Create constant-filled storage with NumPy when worthwhile."""
    if not _numpy_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import full

    return full(shape, fill_value, dtype=dtype)


def execute_eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Create identity-like matrix storage with NumPy."""
    shape = (rows, columns)
    if not _numpy_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import eye

    return eye(rows, columns, k, dtype=dtype)


def execute_arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Create arithmetic-progression storage with NumPy."""
    if not _numpy_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import arange

    return arange(start, step, count, dtype=dtype)


def execute_linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Create evenly spaced storage with NumPy on safe finite endpoints."""
    if not _numpy_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import linspace

    return linspace(start, stop, count, dtype=dtype)


def execute_transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run a storage permutation with NumPy."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import transpose

    return transpose(value, permutation, output_shape=output_shape)


def execute_concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Join tensor storage along an existing axis with NumPy."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import concat

    return concat(values, axis, dtype=dtype, output_shape=output_shape)


def execute_stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Join tensor storage along a new axis with NumPy."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import stack

    return stack(values, axis, dtype=dtype, output_shape=output_shape)


def execute_outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run a vector outer product with NumPy."""
    work = left.size * right.size
    if not _numpy_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    from .numpy import outer

    return outer(left, right, dtype=dtype)


def execute_outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[array[Any], array[Any]] | None:
    """Run outer-product VJPs when native sums preserve semantics."""
    if not _numpy_work_is_large_enough(
        grad.size,
        _NUMPY_MATMUL_MIN_WORK,
    ):
        return None

    from .numpy import outer_gradient

    return outer_gradient(grad, left, right)


def execute_sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> array[Any] | None:
    """Run a fused SGD parameter update."""
    if not _numpy_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import sgd_update

    return sgd_update(parameter, gradient, learning_rate)


def execute_adam_update(
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
) -> tuple[array[Any], array[Any], array[Any], array[Any], array[Any]] | None:
    """Run a fused Adam update on ordinary finite optimizer state."""
    if not _numpy_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import adam_update

    return adam_update(
        parameter,
        gradient,
        moment,
        scale,
        scaled,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_correction=first_correction,
        second_correction=second_correction,
    )


def execute_rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[array[Any], array[Any], array[Any], array[Any]] | None:
    """Run a fused RMSprop update on ordinary finite optimizer state."""
    if not _numpy_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import rmsprop_update

    return rmsprop_update(
        parameter,
        gradient,
        scale,
        scaled,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )


def execute_reduction(
    operation: ReductionOperation,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an accelerated reduction or request the stable Python fallback."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    from .numpy import reduction

    return reduction(
        operation,
        value,
        axes,
        keepdims=keepdims,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_reduction_gradient(
    operation: DifferentiableReductionOperation,
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> array[Any] | None:
    """Run a fused reduction VJP or request the Python fallback."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    from .numpy import reduction_gradient

    return reduction_gradient(
        operation,
        grad,
        value,
        axes,
        keepdims=keepdims,
    )


def execute_arg_extremum(
    operation: ArgExtremumOperation,
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an argmin/argmax reduction using NumPy."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    from .numpy import arg_extremum

    return arg_extremum(
        operation,
        value,
        axis,
        keepdims=keepdims,
        output_shape=output_shape,
    )


def execute_comparison(
    operation: ComparisonOperation,
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an elementwise broadcasting comparison."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import comparison

    return comparison(operation, left, right, output_shape=output_shape)


def execute_where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an elementwise broadcasting selection."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import where

    return where(
        condition,
        left,
        right,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_where_gradient(
    grad: Tensor,
    condition: Tensor,
) -> tuple[array[Any], array[Any]] | None:
    """Split a selection gradient along a condition mask."""
    if not _numpy_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import where_gradient

    return where_gradient(grad, condition)


def execute_clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run an elementwise clipping kernel."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import clip

    return clip(
        value,
        min_value,
        max_value,
        dtype=dtype,
    )


def execute_clip_gradient(
    grad: Tensor,
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
) -> array[Any] | None:
    """Run the clipping VJP with zero boundary subgradients."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import clip_gradient

    return clip_gradient(grad, value, min_value, max_value)


def execute_extremum(
    operation: ExtremumOperation,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run an elementwise broadcasting minimum or maximum."""
    if not _numpy_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import extremum

    return extremum(
        operation,
        left,
        right,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_extremum_gradient(
    operation: ExtremumOperation,
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[array[Any], array[Any]] | None:
    """Split an elementwise-extremum VJP, including tie sharing."""
    if not _numpy_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import extremum_gradient

    return extremum_gradient(operation, grad, left, right)


def execute_normalization(
    operation: NormalizationOperation,
    value: Tensor,
    axis: int,
    *,
    dtype: DataType,
) -> array[Any] | None:
    """Run a fused softmax-family transform on ordinary finite inputs."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import normalization

    return normalization(operation, value, axis, dtype=dtype)


def execute_normalization_gradient(
    operation: NormalizationOperation,
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> array[Any] | None:
    """Run a fused softmax-family VJP when cancellation risk is low."""
    if not _numpy_work_is_large_enough(
        max(grad.size, value.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import normalization_gradient

    return normalization_gradient(operation, grad, value, axis)


def execute_logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run a fused stable log-sum-exp reduction."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    from .numpy import logsumexp

    return logsumexp(
        value,
        axes,
        keepdims=keepdims,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_logsumexp_gradient(
    grad: Tensor,
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
) -> array[Any] | None:
    """Run a fused log-sum-exp VJP on finite inputs."""
    if not _numpy_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    from .numpy import logsumexp_gradient

    return logsumexp_gradient(
        grad,
        value,
        axes,
        keepdims=keepdims,
    )


def execute_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run fused multiclass cross-entropy on broadcast dense targets."""
    if not _numpy_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import cross_entropy

    return cross_entropy(
        logits,
        targets,
        axis,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
) -> tuple[array[Any], array[Any]] | None:
    """Run fused multiclass cross-entropy VJPs when numerically safe."""
    if not _numpy_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import cross_entropy_gradient

    return cross_entropy_gradient(
        grad,
        logits,
        targets,
        axis,
        reduction=reduction,
    )


def execute_binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> array[Any] | None:
    """Run fused binary cross-entropy on broadcast inputs."""
    if not _numpy_work_is_large_enough(
        prediction.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import binary_cross_entropy

    return binary_cross_entropy(
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_binary_cross_entropy_gradient(
    grad: Tensor,
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
) -> tuple[array[Any], array[Any]] | None:
    """Run fused binary cross-entropy VJPs."""
    if not _numpy_work_is_large_enough(
        prediction.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import binary_cross_entropy_gradient

    return binary_cross_entropy_gradient(
        grad,
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
    )


def execute_sum_to_shape(
    gradient: Tensor,
    shape: tuple[int, ...],
) -> array[Any] | None:
    """Reduce broadcast-gradient contributions with NumPy when safe."""
    if not _numpy_work_is_large_enough(
        gradient.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import sum_to_shape

    return sum_to_shape(gradient, shape)


def execute_sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> array[Any] | None:
    """Run a fused NumPy multiply-and-broadcast-reduction when safe."""
    work = max(gradient.size, factor.size)
    if not _numpy_work_is_large_enough(
        work,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import sum_products_to_shape

    return sum_products_to_shape(gradient, factor, shape)


def execute_division_denominator_gradient(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
) -> array[Any] | None:
    """Run the accelerated division-denominator VJP when numerically safe."""
    if not _numpy_work_is_large_enough(
        max(grad.size, numerator.size, denominator.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import division_denominator_gradient

    return division_denominator_gradient(grad, numerator, denominator)


def execute_power_base_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> array[Any] | None:
    """Run the accelerated power-base VJP when numerically safe."""
    if not _numpy_work_is_large_enough(
        max(grad.size, base.size, exponent.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import power_base_gradient

    return power_base_gradient(grad, base, exponent)


def execute_power_exponent_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> array[Any] | None:
    """Run the accelerated power-exponent VJP when numerically safe."""
    if not _numpy_work_is_large_enough(
        max(grad.size, base.size, exponent.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    from .numpy import power_exponent_gradient

    return power_exponent_gradient(grad, base, exponent)


__all__ = [
    "BackendName",
    "BackendSelection",
    "BackendUnavailableError",
    "available_backends",
    "get_backend",
    "set_backend",
    "use_backend",
]
