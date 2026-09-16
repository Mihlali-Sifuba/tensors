"""Reference the standard deviation VJP for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
import math as _math
from tensors.tensor import Tensor
from tensors.math._reduction import reduction_groups
from tensors.math.mean import _stable_float_mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def _scaled_deviations(
    value: Tensor, group: list[int]
) -> tuple[float, list[float], float]:
    """Return a safe scale, centered scaled values, and their deviation."""
    values = [float(value._data[index]) for index in group]
    if any((not _math.isfinite(item) for item in values)):
        return (_math.nan, [_math.nan] * len(values), _math.nan)
    count = len(values)
    average = _stable_float_mean(values)
    centered = [item - average for item in values]
    if all((_math.isfinite(item) for item in centered)):
        scale = max((abs(item) for item in centered), default=0.0)
        if scale == 0.0:
            return (0.0, [0.0] * count, 0.0)
        normalized_centered = [item / scale for item in centered]
    else:
        scale = max((abs(item) for item in values), default=0.0)
        normalized = [item / scale for item in values]
        normalized_average = _stable_float_mean(normalized)
        normalized_centered = [item - normalized_average for item in normalized]
    variance = _math.fsum((item * item / count for item in normalized_centered))
    return (scale, normalized_centered, _math.sqrt(variance))


def reduce_std_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Differentiate the standard deviation of each group."""
    axis = axes
    _, _, groups = reduction_groups(value, axis, keepdims, scalar_as_vector=True)
    result = [0.0] * value.size
    for output_index, group in enumerate(groups):
        if not group:
            continue
        _, centered, normalized_deviation = _scaled_deviations(value, group)
        if normalized_deviation == 0:
            continue
        normalizer = len(group) * normalized_deviation
        upstream = grad._data[output_index]
        for input_index, centered_value in zip(group, centered):
            result[input_index] = upstream * (centered_value / normalizer)
    return PythonStorage.from_values(result, grad.dtype)
