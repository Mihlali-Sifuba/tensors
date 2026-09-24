"""Reference the index of the minimum for the Python backend."""

from __future__ import annotations

import math

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import int64
from tensors.utils.reductions import reduction_groups
from tensors.utils.coordinates import linear_index_to_coordinates


def argmin(
    value_values,
    input_shape: tuple[int, ...],
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the index of the minimum element along the axis.

    A NaN anywhere in a group wins: comparisons against it are all false, so
    it is selected explicitly rather than skipped. Ties keep the first index.
    """
    _, _, groups = reduction_groups(input_shape, axis, keepdims, scalar_as_vector=True)
    if any(not group for group in groups):
        raise ValueError("Cannot compute argmin of empty tensor")
    indices = []
    for group in groups:
        selected = next(
            (
                index
                for index in group
                if isinstance(value_values[index], float)
                and math.isnan(value_values[index])
            ),
            None,
        )
        if selected is None:
            selected = group[0]
            for candidate in group[1:]:
                if value_values[candidate] < value_values[selected]:
                    selected = candidate
        indices.append(
            selected
            if axis is None
            else linear_index_to_coordinates(selected, input_shape)[axis]
        )
    return PythonStorage.from_values(indices, int64)
