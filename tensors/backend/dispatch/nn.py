"""Dispatch for normalization, probability, and loss kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import _NUMPY_ELEMENTWISE_MIN_SIZE, _array_work_is_large_enough
from ..storage import Storage
from ..types import LossReduction, NormalizationOperation

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

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
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested multiclass cross-entropy VJPs when safe."""
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
        needs_input_grad=needs_input_grad,
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
    needs_input_grad: tuple[bool, ...] = (True, True),
) -> tuple[Storage | None, Storage | None] | None:
    """Run the requested binary cross-entropy VJPs."""
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
        needs_input_grad=needs_input_grad,
    )
