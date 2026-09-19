"""Reference the Euclidean norm for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
import math
from tensors.tensor import Tensor
from tensors.utils.reductions import reduction_groups
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType


def _scaled_norm(value: Tensor, group: list[int]) -> tuple[float, list[float], float]:
    """Return a safe scale, scaled values, and their Euclidean norm."""
    values = [float(value._data[index]) for index in group]
    if any((math.isinf(item) for item in values)):
        return (1.0, [math.nan] * len(values), math.inf)
    if any((math.isnan(item) for item in values)):
        return (1.0, [math.nan] * len(values), math.nan)
    scale = max((abs(item) for item in values), default=0.0)
    if scale == 0.0:
        return (0.0, [0.0] * len(values), 0.0)
    normalized = [item / scale for item in values]
    normalized_magnitude = math.sqrt(math.fsum((item * item for item in normalized)))
    return (scale, normalized, normalized_magnitude)


def reduce_norm(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return the Euclidean norm of each group without overflow."""
    axis = axes
    _, output_shape, groups = reduction_groups(value.shape, axis, keepdims)
    results = []
    for group in groups:
        scale, _, normalized_magnitude = _scaled_norm(value, group)
        results.append(scale * normalized_magnitude)
    return PythonStorage.from_values(results, dtype)
