"""Dispatch for normalization, probability, and loss kernels."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.loading import _backend_kernel
from tensors.backend.policy import (
    _NUMPY_ELEMENTWISE_MIN_SIZE,
    _array_work_is_large_enough,
)
from tensors.backend.storage import Storage
from tensors.backend.types import LossReduction

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run fused multiclass cross-entropy on broadcast dense targets."""
    from tensors.backend.python.kernels.nn.cross_entropy import (
        cross_entropy as reference,
    )

    if not _array_work_is_large_enough(logits.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            logits,
            targets,
            axis,
            reduction=reduction,
            dtype=dtype,
            output_shape=output_shape,
        )
    cross_entropy = _backend_kernel("cross_entropy")
    result = cross_entropy(
        logits,
        targets,
        axis,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is not None:
        return result
    return reference(
        logits,
        targets,
        axis,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )
