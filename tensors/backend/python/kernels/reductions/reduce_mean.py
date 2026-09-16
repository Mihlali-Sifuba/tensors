"""Reference the arithmetic mean for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math
from tensors.math._reduction import reduction_groups
from tensors.math.sum import _stable_float_sum, _sum_exact_ratios


def _stable_float_mean(values: list[float]) -> float:
    """Return a mean without overflowing its sum or underflowing its terms."""
    if not values:
        return math.nan
    if any((not math.isfinite(value) for value in values)):
        return _stable_float_sum(values) / len(values)
    return _sum_exact_ratios(
        [value.as_integer_ratio() for value in values], divisor=len(values)
    )


def reduce_mean(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the arithmetic mean of each group."""
    data = value._data
    if axes == tuple(range(value.ndim)):
        return PythonStorage.from_values(
            [_stable_float_mean([float(value) for value in data])], dtype
        )
    _, output_shape, groups = reduction_groups(
        value, axes, keepdims, scalar_as_vector=True
    )
    values = [
        _stable_float_mean([float(data[index]) for index in group]) for group in groups
    ]
    return PythonStorage.from_values(values, dtype)
