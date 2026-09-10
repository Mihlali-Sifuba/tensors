"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _NUMPY_REDUCTION_MIN_SIZE,
    _array_work_is_large_enough,
)
from ..storage import Storage
from ..types import (
    ArgExtremumOperation,
    DifferentiableReductionOperation,
    ReductionOperation,
)

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

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
