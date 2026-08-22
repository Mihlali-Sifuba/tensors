"""Numerical backend selection and internal kernel dispatch."""

from __future__ import annotations

import importlib.util
import os
import threading
from array import array
from collections.abc import Iterator
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
    "norm",
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
