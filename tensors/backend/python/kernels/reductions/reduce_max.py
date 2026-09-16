"""Reference the maximum for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import builtins
import math
from tensors.utils.reductions import reduction_groups


def reduce_max(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the largest element of each group."""
    _, output_shape, groups = reduction_groups(
        value.shape, axes, keepdims, scalar_as_vector=True
    )
    if any((not group for group in groups)):
        raise ValueError("Cannot compute max of empty tensor")
    values = []
    for group in groups:
        group_values = [value._data[index] for index in group]
        if any((isinstance(item, float) and math.isnan(item) for item in group_values)):
            values.append(math.nan)
        else:
            values.append(builtins.max(group_values))
    return PythonStorage.from_values(values, dtype)
