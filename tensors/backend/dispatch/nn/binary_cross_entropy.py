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


def execute_binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run fused binary cross-entropy on broadcast inputs."""
    from tensors.backend.python.kernels.nn.binary_cross_entropy import (
        binary_cross_entropy as reference,
    )

    if not _array_work_is_large_enough(prediction.size, _NUMPY_ELEMENTWISE_MIN_SIZE):
        return reference(
            prediction,
            target,
            from_logits=from_logits,
            reduction=reduction,
            dtype=dtype,
            output_shape=output_shape,
        )
    binary_cross_entropy = _backend_kernel("binary_cross_entropy")
    result = binary_cross_entropy(
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is not None:
        return result
    return reference(
        prediction,
        target,
        from_logits=from_logits,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )
