"""NumPy implementation of the RMSprop parameter update."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from typing import cast
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage] | None:
    """Apply one fused RMSprop update on finite optimizer state."""
    tensors = (parameter, gradient, scale, scaled)
    values = [_view(item).astype(numpy.float64, copy=False) for item in tensors]
    parameter_values, gradients, scales, scaled_values = values
    if not _finite_operands(*values):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        new_scales = numpy.maximum(scales, numpy.abs(gradients))
        safe_scales = numpy.where(new_scales == 0.0, 1.0, new_scales)
        previous_ratio = scales / safe_scales
        gradient_ratio = numpy.abs(gradients) / safe_scales
        new_scaled = (
            rho * scaled_values * previous_ratio * previous_ratio
            + (1.0 - rho) * gradient_ratio * gradient_ratio
        )
        new_scaled = numpy.where(new_scales == 0.0, 0.0, new_scaled)
        root_moment = new_scales * numpy.sqrt(new_scaled)
        parameter_result = parameter_values - learning_rate * gradients / (
            root_moment + epsilon
        )
    if not _finite_operands(new_scales, new_scaled, parameter_result):
        return None
    specifications = (
        (parameter_result, parameter.dtype),
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
    return cast("tuple[Storage, Storage, Storage]", storages)
