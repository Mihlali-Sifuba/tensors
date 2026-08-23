"""Numerical backend selection and internal kernel dispatch."""

from __future__ import annotations

import importlib
import importlib.util
import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from ..storage import Storage

if TYPE_CHECKING:
    from .._typing import Scalar, TensorIndex
    from ..dtype import DataType
    from ..tensor import Tensor


BackendName: TypeAlias = Literal["python", "numpy", "cuda"]
BackendSelection: TypeAlias = Literal["python", "numpy", "cuda", "auto"]
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
    "sum",
    "mean",
    "variance",
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


_VALID_BACKENDS = {"python", "numpy", "cuda", "auto"}
_NUMPY_ELEMENTWISE_MIN_SIZE = 32
_NUMPY_REDUCTION_MIN_SIZE = 8
_NUMPY_MATMUL_MIN_WORK = 32
_CUDA_FUSION_MIN_WORK = 8_192
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


def _cuda_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""
    try:
        if importlib.util.find_spec("cupy") is None:
            return False
        cupy = importlib.import_module("cupy")
        return bool(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        # Import, driver, and runtime errors all mean the backend cannot execute.
        return False


def available_backends() -> tuple[BackendName, ...]:
    """Return the numerical backends available in this environment."""
    available: list[BackendName] = ["python"]
    if _numpy_available():
        available.append("numpy")
    if _cuda_available():
        available.append("cuda")
    return tuple(available)


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
    if normalized == "cuda" and not _cuda_available():
        raise BackendUnavailableError(
            "The CUDA backend is unavailable. Install the CuPy build matching "
            "your driver with `pip install \"ms-tensors[cuda12]\"` or "
            "`pip install \"ms-tensors[cuda13]\"`."
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


def _array_work_is_large_enough(work: int, minimum: int) -> bool:
    backend = get_backend()
    if backend == "cuda":
        # Explicit CUDA selection keeps supported operations on-device. Kernel
        # implementations can still request the Python fallback when needed.
        return True
    return backend == "numpy" and work >= minimum


@lru_cache(maxsize=None)
def _load_backend_kernel(backend: BackendName, name: str) -> Any:
    """Load and cache one optional-backend kernel callable."""
    if backend not in {"numpy", "cuda"}:
        raise RuntimeError("A kernel was requested for the Python backend")
    module = importlib.import_module(f"{__name__}.{backend}")
    return getattr(module, name)


def _backend_kernel(name: str) -> Any:
    """Return a cached kernel from the active optional backend."""
    return _load_backend_kernel(get_backend(), name)


def _clear_backend_kernel_cache() -> None:
    """Forget cached callables after an explicit backend-context change."""
    _load_backend_kernel.cache_clear()


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
        _clear_backend_kernel_cache()


@contextmanager
def use_backend(backend: BackendSelection) -> Iterator[None]:
    """Temporarily select a backend in the current execution context."""
    selected = _resolve_backend(backend)
    # Besides keeping the cache bounded across backend changes, clearing here
    # ensures deliberate runtime replacement of an internal kernel (for
    # instrumentation or tests) is observed on entry to the scoped backend.
    _clear_backend_kernel_cache()
    token = _backend_override.set(selected)
    try:
        yield
    finally:
        _backend_override.reset(token)
        _clear_backend_kernel_cache()


def execute_matmul(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run an accelerated matrix product or request the Python fallback."""
    contraction_size = left.shape[-1]
    work = _shape_size(output_shape) * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    matmul = _backend_kernel("matmul")
    return matmul(left, right, dtype=dtype, output_shape=output_shape)


def execute_matmul_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Run matrix-product VJPs with an accelerated backend when safe."""
    contraction_size = left.shape[-1]
    work = grad.size * contraction_size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    matmul_gradient = _backend_kernel("matmul_gradient")
    return matmul_gradient(grad, left, right)


def execute_binary(
    operation: BinaryOperation,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run an accelerated binary operation or request the Python fallback."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    binary = _backend_kernel("binary")
    return binary(
        operation,
        left,
        right,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_fused_elementwise(
    value: Tensor,
    steps: Sequence[tuple[str, float | None, bool]],
    *,
    dtype: DataType,
) -> tuple[Storage, ...] | None:
    """Run a compatible scalar expression chain in one CUDA kernel."""
    if (
        get_backend() != "cuda"
        or len(steps) < 2
        or dtype.name != "float64"
        or (
            value.size * len(steps) < _CUDA_FUSION_MIN_WORK
            and len(steps) < 64
        )
    ):
        return None
    fused_elementwise = _backend_kernel("fused_elementwise")
    return fused_elementwise(value, tuple(steps), dtype=dtype)


def execute_negate(
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run accelerated elementwise negation or request the Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    negate = _backend_kernel("negate")
    return negate(value, dtype=dtype)


def execute_unary(
    operation: UnaryOperation,
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run an accelerated unary transform or request the Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    unary = _backend_kernel("unary")
    return unary(operation, value, dtype=dtype)


def execute_unary_gradient(
    operation: UnaryOperation,
    grad: Tensor,
    value: Tensor,
) -> Storage | None:
    """Run an accelerated unary VJP or request the Python fallback."""
    if not _array_work_is_large_enough(
        max(grad.size, value.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    unary_gradient = _backend_kernel("unary_gradient")
    return unary_gradient(operation, grad, value)


def execute_slice(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run accelerated tensor slicing or request the Python fallback."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    slice_tensor = _backend_kernel("slice_tensor")
    return slice_tensor(value, key, output_shape=output_shape)


def execute_slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run accelerated slice scattering or request the Python fallback."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    slice_scatter = _backend_kernel("slice_scatter")
    return slice_scatter(value, indices, output_shape=output_shape)


def execute_cast(
    value: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run accelerated dtype conversion or request the Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    cast_tensor = _backend_kernel("cast_tensor")
    return cast_tensor(value, dtype=dtype)


def execute_full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create constant-filled storage with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    full = _backend_kernel("full")
    return full(shape, fill_value, dtype=dtype)


def execute_eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create identity-like matrix storage with an accelerated backend."""
    shape = (rows, columns)
    if not _array_work_is_large_enough(
        _shape_size(shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    eye = _backend_kernel("eye")
    return eye(rows, columns, k, dtype=dtype)


def execute_arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create arithmetic-progression storage with an accelerated backend."""
    if not _array_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    arange = _backend_kernel("arange")
    return arange(start, step, count, dtype=dtype)


def execute_linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create evenly spaced storage with an accelerated backend when safe."""
    if not _array_work_is_large_enough(
        count,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    linspace = _backend_kernel("linspace")
    return linspace(start, stop, count, dtype=dtype)


def execute_transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a storage permutation with an accelerated backend."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    transpose = _backend_kernel("transpose")
    return transpose(value, permutation, output_shape=output_shape)


def execute_concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Join tensor storage along an existing axis with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    concat = _backend_kernel("concat")
    return concat(values, axis, dtype=dtype, output_shape=output_shape)


def execute_stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Join tensor storage along a new axis with an accelerated backend."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    stack = _backend_kernel("stack")
    return stack(values, axis, dtype=dtype, output_shape=output_shape)


def execute_outer(
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a vector outer product with an accelerated backend."""
    work = left.size * right.size
    if not _array_work_is_large_enough(work, _NUMPY_MATMUL_MIN_WORK):
        return None

    outer = _backend_kernel("outer")
    return outer(left, right, dtype=dtype)


def execute_outer_gradient(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Storage, Storage] | None:
    """Run outer-product VJPs when native sums preserve semantics."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_MATMUL_MIN_WORK,
    ):
        return None

    outer_gradient = _backend_kernel("outer_gradient")
    return outer_gradient(grad, left, right)


def execute_sgd_update(
    parameter: Tensor,
    gradient: Tensor,
    learning_rate: float,
) -> Storage | None:
    """Run a fused SGD parameter update."""
    if not _array_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    sgd_update = _backend_kernel("sgd_update")
    return sgd_update(parameter, gradient, learning_rate)


def execute_sgd_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    learning_rate: float,
) -> tuple[Storage, ...] | None:
    """Update several compatible parameters with one native array batch."""
    if (
        len(parameters) < 2
        or len(parameters) != len(gradients)
        or not _array_work_is_large_enough(
            sum(parameter.size for parameter in parameters),
            _NUMPY_ELEMENTWISE_MIN_SIZE,
        )
    ):
        return None
    sgd_updates = _backend_kernel("sgd_updates")
    return sgd_updates(parameters, gradients, learning_rate)


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
) -> tuple[Storage, Storage, Storage, Storage, Storage] | None:
    """Run a fused Adam update on ordinary finite optimizer state."""
    if not _array_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    adam_update = _backend_kernel("adam_update")
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


def execute_adam_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    moments: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_corrections: Sequence[float],
    second_corrections: Sequence[float],
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible Adam states in one native array batch."""
    count = len(parameters)
    if (
        count < 2
        or any(
            len(items) != count
            for items in (
                gradients,
                moments,
                scales,
                scaled_values,
                first_corrections,
                second_corrections,
            )
        )
        or not _array_work_is_large_enough(
            sum(parameter.size for parameter in parameters),
            _NUMPY_ELEMENTWISE_MIN_SIZE,
        )
    ):
        return None
    adam_updates = _backend_kernel("adam_updates")
    return adam_updates(
        parameters,
        gradients,
        moments,
        scales,
        scaled_values,
        beta1=beta1,
        beta2=beta2,
        learning_rate=learning_rate,
        epsilon=epsilon,
        first_corrections=first_corrections,
        second_corrections=second_corrections,
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
) -> tuple[Storage, Storage, Storage, Storage] | None:
    """Run a fused RMSprop update on ordinary finite optimizer state."""
    if not _array_work_is_large_enough(
        parameter.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    rmsprop_update = _backend_kernel("rmsprop_update")
    return rmsprop_update(
        parameter,
        gradient,
        scale,
        scaled,
        rho=rho,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )


def execute_rmsprop_updates(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    scales: Sequence[Tensor],
    scaled_values: Sequence[Tensor],
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[tuple[Storage, ...], ...] | None:
    """Update several compatible RMSprop states in one native array batch."""
    count = len(parameters)
    if (
        count < 2
        or any(
            len(items) != count
            for items in (gradients, scales, scaled_values)
        )
        or not _array_work_is_large_enough(
            sum(parameter.size for parameter in parameters),
            _NUMPY_ELEMENTWISE_MIN_SIZE,
        )
    ):
        return None
    rmsprop_updates = _backend_kernel("rmsprop_updates")
    return rmsprop_updates(
        parameters,
        gradients,
        scales,
        scaled_values,
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
) -> Storage | None:
    """Run an accelerated reduction or request the stable Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    reduction = _backend_kernel("reduction")
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
) -> Storage | None:
    """Run a fused reduction VJP or request the Python fallback."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    reduction_gradient = _backend_kernel("reduction_gradient")
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
) -> Storage | None:
    """Run an argmin/argmax reduction with an accelerated backend."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    arg_extremum = _backend_kernel("arg_extremum")
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
) -> Storage | None:
    """Run an elementwise broadcasting comparison."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    comparison = _backend_kernel("comparison")
    return comparison(operation, left, right, output_shape=output_shape)


def execute_where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run an elementwise broadcasting selection."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    where = _backend_kernel("where")
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
) -> tuple[Storage, Storage] | None:
    """Split a selection gradient along a condition mask."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    where_gradient = _backend_kernel("where_gradient")
    return where_gradient(grad, condition)


def execute_clip(
    value: Tensor,
    min_value: int | float | None,
    max_value: int | float | None,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run an elementwise clipping kernel."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    clip = _backend_kernel("clip")
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
) -> Storage | None:
    """Run the clipping VJP with zero boundary subgradients."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    clip_gradient = _backend_kernel("clip_gradient")
    return clip_gradient(grad, value, min_value, max_value)


def execute_extremum(
    operation: ExtremumOperation,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run an elementwise broadcasting minimum or maximum."""
    if not _array_work_is_large_enough(
        _shape_size(output_shape),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    extremum = _backend_kernel("extremum")
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
) -> tuple[Storage, Storage] | None:
    """Split an elementwise-extremum VJP, including tie sharing."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    extremum_gradient = _backend_kernel("extremum_gradient")
    return extremum_gradient(operation, grad, left, right)


def execute_normalization(
    operation: NormalizationOperation,
    value: Tensor,
    axis: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Run a fused softmax-family transform on ordinary finite inputs."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    normalization = _backend_kernel("normalization")
    return normalization(operation, value, axis, dtype=dtype)


def execute_normalization_gradient(
    operation: NormalizationOperation,
    grad: Tensor,
    value: Tensor,
    axis: int,
) -> Storage | None:
    """Run a fused softmax-family VJP when cancellation risk is low."""
    if not _array_work_is_large_enough(
        max(grad.size, value.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    normalization_gradient = _backend_kernel("normalization_gradient")
    return normalization_gradient(operation, grad, value, axis)


def execute_logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a fused stable log-sum-exp reduction."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    logsumexp = _backend_kernel("logsumexp")
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
) -> Storage | None:
    """Run a fused log-sum-exp VJP on finite inputs."""
    if not _array_work_is_large_enough(
        value.size,
        _NUMPY_REDUCTION_MIN_SIZE,
    ):
        return None

    logsumexp_gradient = _backend_kernel("logsumexp_gradient")
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
) -> Storage | None:
    """Run fused multiclass cross-entropy on broadcast dense targets."""
    if not _array_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    cross_entropy = _backend_kernel("cross_entropy")
    return cross_entropy(
        logits,
        targets,
        axis,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )


def execute_one_hot_targets(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> Storage | None:
    """Expand class-index targets with the active array backend."""
    if not _array_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    one_hot_targets = _backend_kernel("one_hot_targets")
    return one_hot_targets(logits, targets, axis)


def execute_validate_distributions(
    targets: Tensor,
    axis: int,
) -> bool | None:
    """Validate dense probability rows with the active array backend."""
    if not _array_work_is_large_enough(
        targets.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    distributions_valid = _backend_kernel("distributions_valid")
    return distributions_valid(targets, axis)


def execute_cross_entropy_gradient(
    grad: Tensor,
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
) -> tuple[Storage, Storage] | None:
    """Run fused multiclass cross-entropy VJPs when numerically safe."""
    if not _array_work_is_large_enough(
        logits.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    cross_entropy_gradient = _backend_kernel("cross_entropy_gradient")
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
) -> Storage | None:
    """Run fused binary cross-entropy on broadcast inputs."""
    if not _array_work_is_large_enough(
        prediction.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    binary_cross_entropy = _backend_kernel("binary_cross_entropy")
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
) -> tuple[Storage, Storage] | None:
    """Run fused binary cross-entropy VJPs."""
    if not _array_work_is_large_enough(
        prediction.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    binary_cross_entropy_gradient = _backend_kernel("binary_cross_entropy_gradient")
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
) -> Storage | None:
    """Reduce broadcast-gradient contributions with an accelerated backend."""
    if not _array_work_is_large_enough(
        gradient.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    sum_to_shape = _backend_kernel("sum_to_shape")
    return sum_to_shape(gradient, shape)


def execute_sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Run a fused accelerated multiply-and-broadcast reduction when safe."""
    work = max(gradient.size, factor.size)
    if not _array_work_is_large_enough(
        work,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    sum_products_to_shape = _backend_kernel("sum_products_to_shape")
    return sum_products_to_shape(gradient, factor, shape)


def execute_division_denominator_gradient(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
) -> Storage | None:
    """Run the accelerated division-denominator VJP when numerically safe."""
    if not _array_work_is_large_enough(
        max(grad.size, numerator.size, denominator.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    division_denominator_gradient = _backend_kernel("division_denominator_gradient")
    return division_denominator_gradient(grad, numerator, denominator)


def execute_power_base_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Run the accelerated power-base VJP when numerically safe."""
    if not _array_work_is_large_enough(
        max(grad.size, base.size, exponent.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    power_base_gradient = _backend_kernel("power_base_gradient")
    return power_base_gradient(grad, base, exponent)


def execute_power_exponent_gradient(
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
) -> Storage | None:
    """Run the accelerated power-exponent VJP when numerically safe."""
    if not _array_work_is_large_enough(
        max(grad.size, base.size, exponent.size),
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    power_exponent_gradient = _backend_kernel("power_exponent_gradient")
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
