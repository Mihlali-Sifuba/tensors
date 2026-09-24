"""Reference the standard deviation VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
from tensors.utils.deviation import scaled_deviations
from tensors.utils.reductions import reduction_groups


def reduce_std_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Differentiate the standard deviation of each group."""
    axis = axes
    _, _, groups = reduction_groups(input_shape, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        if not group:
            continue
        _, centered, normalized_deviation = scaled_deviations(value_values, group)
        if normalized_deviation == 0:
            continue
        normalizer = len(group) * normalized_deviation
        upstream = grad_values[output_index]
        for input_index, centered_value in zip(group, centered):
            result[input_index] = upstream * (centered_value / normalizer)
    return PythonStorage.from_values(result, dtype)
