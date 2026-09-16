"""Reference the variance VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.math._reduction import reduction_groups
from tensors.math.std import _scaled_deviations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def reduce_variance_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Differentiate the variance of each group."""
    axis = axes
    _, _, groups = reduction_groups(value, axis, keepdims, scalar_as_vector=True)
    gradients = [0.0] * value.size
    for output_index, group in enumerate(groups):
        if not group:
            continue
        upstream = grad._data[output_index]
        if upstream == 0:
            continue
        scale, centered, _ = _scaled_deviations(value, group)
        if all((centered_value == 0.0 for centered_value in centered)):
            continue
        factor = scale * (2.0 / len(group))
        for input_index, centered_value in zip(group, centered):
            gradients[input_index] = upstream * centered_value * factor
    return PythonStorage.from_values(gradients, grad.dtype)
