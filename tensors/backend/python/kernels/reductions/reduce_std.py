"""Reference the standard deviation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math as _math
from tensors.utils.deviation import scaled_deviations
from tensors.utils.reductions import reduction_groups


def reduce_std(
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the standard deviation of each group."""
    _, output_shape, groups = reduction_groups(
        input_shape, axes, keepdims, scalar_as_vector=True
    )
    values = []
    for group in groups:
        if not group:
            values.append(_math.nan)
            continue
        scale, _, normalized_deviation = scaled_deviations(value_values, group)
        values.append(scale * normalized_deviation)
    return PythonStorage.from_values(values, dtype)
