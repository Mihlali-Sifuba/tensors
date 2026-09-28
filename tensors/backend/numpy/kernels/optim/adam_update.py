"""NumPy implementation of the Adam parameter update."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def adam_update(
    parameter,
    gradient,
    moment,
    scale,
    scaled,
    *,
    beta1: float,
    beta2: float,
    learning_rate: float,
    epsilon: float,
    first_correction: float,
    second_correction: float,
    dtype: DataType,
    shape: tuple[int, ...],
) -> tuple[Storage, Storage, Storage, Storage, Storage] | None:
    """Apply one fused Adam update on finite optimizer state."""
    arrays = (parameter, gradient, moment, scale, scaled)
    values = [numpy.asarray(item).astype(numpy.float64, copy=False) for item in arrays]
    parameter_values, gradients, moments, scales, scaled_values = values
    if not _finite_operands(*values):
        return None
    left_term = beta1 * moments
    right_term = (1.0 - beta1) * gradients
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        new_moments = left_term + right_term
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            beta2 * scaled_values * previous_ratio * previous_ratio
            + (1.0 - beta2) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_correction = numpy.sqrt(second_correction)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        denominator = first_correction * (root_moment + epsilon * root_correction)
        ratio = new_moments * root_correction / denominator
        parameter_result = parameter_values - learning_rate * ratio
        visible = new_scales * new_scales * new_scaled
    if not _finite_operands(new_moments, new_scales, new_scaled, parameter_result):
        return None
    specifications = (
        (parameter_result, dtype),
        (new_moments, dtype),
        (visible, dtype),
        (new_scales, dtype),
        (new_scaled, dtype),
    )
    storages = tuple(
        (
            _storage(result, dtype=result_dtype, output_shape=shape)
            for result, result_dtype in specifications
        )
    )
    if any((storage is None for storage in storages)):
        return None
    return cast("tuple[Storage, Storage, Storage, Storage, Storage]", storages)
