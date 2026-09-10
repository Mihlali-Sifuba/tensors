"""Dispatch for elementwise operations and their VJPs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
    _shape_size,
)
from ..storage import Storage
from ..types import (
    BinaryOperation,
    ComparisonOperation,
    ExtremumOperation,
    UnaryOperation,
)

if TYPE_CHECKING:
    from ..._typing import Scalar
    from ...dtype import DataType
    from ...tensor import Tensor

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
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split a selection gradient along a condition mask."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    where_gradient = _backend_kernel("where_gradient")
    return where_gradient(
        grad,
        condition,
        needs_input_grad=needs_input_grad,
    )

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
    *,
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Split an elementwise-extremum VJP, including tie sharing."""
    if not _array_work_is_large_enough(
        grad.size,
        _NUMPY_ELEMENTWISE_MIN_SIZE,
    ):
        return None

    extremum_gradient = _backend_kernel("extremum_gradient")
    return extremum_gradient(
        operation,
        grad,
        left,
        right,
        needs_input_grad=needs_input_grad,
    )

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
