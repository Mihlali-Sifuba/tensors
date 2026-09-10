"""Dispatch for optimizer updates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..loading import _backend_kernel
from ..policy import _NUMPY_ELEMENTWISE_MIN_SIZE, _array_work_is_large_enough
from ..storage import Storage

if TYPE_CHECKING:
    from ...tensor import Tensor

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
) -> tuple[Storage, Storage, Storage] | None:
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
