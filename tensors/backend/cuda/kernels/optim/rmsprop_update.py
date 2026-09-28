"""CuPy implementation of the RMSprop parameter update."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def rmsprop_update(
    parameter,
    gradient,
    scale,
    scaled,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
    dtype: DataType,
    shape: tuple[int, ...],
) -> tuple[Storage, Storage, Storage] | None:
    """Apply one fused RMSprop update on finite optimizer state."""
    arrays = (parameter, gradient, scale, scaled)
    values = [_widen(cupy.asarray(item)) for item in arrays]
    parameter_values, gradients, scales, scaled_values = values
    if not _finite_operands(*values):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        new_scales = cupy.maximum(scales, cupy.abs(gradients))
        safe_scales = cupy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = cupy.abs(gradients) / safe_scales
        new_scaled = (
            rho * scaled_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = cupy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * cupy.sqrt(new_scaled)
        parameter_result = parameter_values - learning_rate * gradients / (
            root_moment + epsilon
        )
    if not _finite_operands(new_scales, new_scaled, parameter_result):
        return None
    specifications = (
        (parameter_result, dtype),
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
    return cast("tuple[Storage, Storage, Storage]", storages)
