"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_REDUCTION_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_logsumexp(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run a fused stable log-sum-exp reduction."""
    from tensors.backend.python.kernels.reductions.logsumexp import (
        logsumexp as reference,
    )

    if not _array_work_is_large_enough(value.size, _NUMPY_REDUCTION_MIN_SIZE):
        return reference(
            value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
        )
    logsumexp = _backend_kernel("logsumexp")
    result = logsumexp(
        value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
    )
    if result is not None:
        return result
    return reference(
        value, axes, keepdims=keepdims, dtype=dtype, output_shape=output_shape
    )
