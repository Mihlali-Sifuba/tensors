"""Reference the variance for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import math
from tensors.math._reduction import reduction_groups
from tensors.math.std import _scaled_deviations


def reduce_variance(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the variance of each group."""
    _, output_shape, groups = reduction_groups(
        value, axes, keepdims, scalar_as_vector=True
    )
    values = []
    for group in groups:
        if not group:
            values.append(math.nan)
            continue
        scale, _, normalized_deviation = _scaled_deviations(value, group)
        deviation = scale * normalized_deviation
        values.append(deviation * deviation)
    return PythonStorage.from_values(values, dtype)
