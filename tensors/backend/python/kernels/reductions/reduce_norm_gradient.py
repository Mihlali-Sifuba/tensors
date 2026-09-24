"""Reference the Euclidean norm VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
import math
from tensors.utils.deviation import scaled_deviations
from tensors.utils.reductions import reduction_groups


def reduce_norm_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Differentiate the Euclidean norm of each group."""
    _, _, groups = reduction_groups(input_shape, axes, keepdims)
    result = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        group_values = [float(value_values[index]) for index in group]
        if any(math.isinf(item) for item in group_values):
            derivative = [math.nan] * len(group)
        elif any(math.isnan(item) for item in group_values):
            derivative = [math.nan] * len(group)
        else:
            scale = max((abs(item) for item in group_values), default=0.0)
            if scale == 0.0:
                continue
            normalized = [item / scale for item in group_values]
            magnitude = math.sqrt(math.fsum(item * item for item in normalized))
            derivative = [item / magnitude for item in normalized]
        upstream = grad_values[output_index]
        for input_index, factor in zip(group, derivative):
            result[input_index] = upstream * factor
    return PythonStorage.from_values(result, dtype)
