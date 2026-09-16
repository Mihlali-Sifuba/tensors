"""Reference the index of the maximum for the Python backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import int64
from tensors.math._reduction import reduction_groups
from tensors.utils.coordinates import linear_index_to_coordinates

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def argmax(
    value: Tensor,
    axis: int | None,
    *,
    keepdims: bool,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the index of the maximum element along the axis.

    A NaN anywhere in a group wins: comparisons against it are all false, so
    it is selected explicitly rather than skipped. Ties keep the first index.
    """
    _, _, groups = reduction_groups(value, axis, keepdims, scalar_as_vector=True)
    if any(not group for group in groups):
        raise ValueError("Cannot compute argmax of empty tensor")
    indices = []
    for group in groups:
        selected = next(
            (
                index
                for index in group
                if isinstance(value._data[index], float)
                and math.isnan(value._data[index])
            ),
            None,
        )
        if selected is None:
            selected = group[0]
            for candidate in group[1:]:
                if value._data[candidate] > value._data[selected]:
                    selected = candidate
        indices.append(
            selected
            if axis is None
            else linear_index_to_coordinates(selected, value.shape)[axis]
        )
    return PythonStorage.from_values(indices, int64)
