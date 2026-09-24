"""Reference the variance VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
from tensors.utils.reductions import reduction_groups
from tensors.utils.deviation import scaled_deviations


def reduce_variance_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Differentiate the variance of each group."""
    axis = axes
    _, _, groups = reduction_groups(input_shape, axis, keepdims, scalar_as_vector=True)
    gradients = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        if not group:
            continue
        upstream = grad_values[output_index]
        if upstream == 0:
            continue
        scale, centered, _ = scaled_deviations(value_values, group)
        if all((centered_value == 0.0 for centered_value in centered)):
            continue
        factor = scale * (2.0 / len(group))
        for input_index, centered_value in zip(group, centered):
            gradients[input_index] = upstream * centered_value * factor
    return PythonStorage.from_values(gradients, dtype)
