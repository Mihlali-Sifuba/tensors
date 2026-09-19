"""CuPy implementation of the Adam parameter update."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def adam_update(
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
    """Apply one fused Adam update on finite optimizer state."""
    tensors = (parameter, gradient, moment, scale, scaled)
    values = [_working_values(item) for item in tensors]
    parameter_values, gradients, moments, scales, scaled_values = values
    if not _finite_operands(*values):
        return None
    left_term = beta1 * moments
    right_term = (1.0 - beta1) * gradients
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        new_moments = left_term + right_term
        new_scales = cupy.maximum(scales, cupy.abs(gradients))
        safe_scales = cupy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = cupy.abs(gradients) / safe_scales
        new_scaled = (
            beta2 * scaled_values * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        )
        new_scaled = cupy.where(new_scales == 0.0, 0.0, new_scaled)
        root_correction = cupy.sqrt(second_correction)
        root_moment = new_scales * cupy.sqrt(new_scaled)
        denominator = first_correction * (root_moment + epsilon * root_correction)
        ratio = new_moments * root_correction / denominator
        parameter_result = parameter_values - learning_rate * ratio
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(new_moments, new_scales, new_scaled, parameter_result):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
        (new_moments, gradient.dtype),
        (visible, gradient.dtype),
        (new_scales, gradient.dtype),
        (new_scaled, gradient.dtype),
    )
    storages = tuple(
        (
            _storage(result, dtype=dtype, output_shape=gradient.shape)
            for result, dtype in specifications
        )
    )
    if any((storage is None for storage in storages)):
        return None
    return cast("tuple[Storage, Storage, Storage, Storage, Storage]", storages)
