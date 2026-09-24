"""Reference the arithmetic mean for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType

import math
from tensors.utils.reductions import reduction_groups
from tensors.utils.summation import stable_float_sum, sum_exact_ratios


def stable_float_mean(values: list[float]) -> float:
    """Return a mean without overflowing its sum or underflowing its terms."""
    if not values:
        return math.nan
    if any((not math.isfinite(value) for value in values)):
        return stable_float_sum(values) / len(values)
    return sum_exact_ratios(
        [value.as_integer_ratio() for value in values], divisor=len(values)
    )


def reduce_mean(
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the arithmetic mean of each group."""
    data = value_values
    if axes == tuple(range(len(input_shape))):
        return PythonStorage.from_values(
            [stable_float_mean([float(value) for value in data])], dtype
        )
    _, output_shape, groups = reduction_groups(
        input_shape, axes, keepdims, scalar_as_vector=True
    )
    values = [
        stable_float_mean([float(data[index]) for index in group]) for group in groups
    ]
    return PythonStorage.from_values(values, dtype)
