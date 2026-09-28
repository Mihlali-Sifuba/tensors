"""Reference the minimum VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
import builtins
import math
from tensors.utils.reductions import reduction_groups


def reduce_min_gradient(
    grad_values,
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Storage | None:
    """Route the upstream gradient to each group's minimum."""
    axis = axes
    _, _, groups = reduction_groups(input_shape, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * len(value_values)
    for output_index, group in enumerate(groups):
        if any(
            (
                isinstance(value_values[index], float)
                and math.isnan(value_values[index])
                for index in group
            )
        ):
            for input_index in group:
                result[input_index] = math.nan
            continue
        minimum = builtins.min((value_values[index] for index in group))
        selected = [index for index in group if value_values[index] == minimum]
        share = grad_values[output_index] / len(selected)
        for input_index in selected:
            result[input_index] = share
    return PythonStorage.from_values(result, dtype)
