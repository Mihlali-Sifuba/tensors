"""Reference the variance for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType

import math
from tensors.utils.reductions import reduction_groups
from tensors.utils.deviation import scaled_deviations


def reduce_variance(
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the variance of each group."""
    _, output_shape, groups = reduction_groups(
        input_shape, axes, keepdims, scalar_as_vector=True
    )
    values = []
    for group in groups:
        if not group:
            values.append(math.nan)
            continue
        scale, _, normalized_deviation = scaled_deviations(value_values, group)
        deviation = scale * normalized_deviation
        values.append(deviation * deviation)
    return PythonStorage.from_values(values, dtype)
