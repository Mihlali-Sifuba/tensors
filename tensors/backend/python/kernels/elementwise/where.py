"""Reference elementwise selection for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.broadcasting import broadcast_to
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def where(
    condition: Tensor,
    left: Tensor,
    right: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Select elementwise from two operands by a condition mask."""
    shape = output_shape
    expanded_condition = broadcast_to(condition, shape)
    expanded_left = broadcast_to(left, shape)
    expanded_right = broadcast_to(right, shape)
    return PythonStorage.from_values(
        [
            left_value if selected != 0 else right_value
            for selected, left_value, right_value in zip(
                expanded_condition._data, expanded_left._data, expanded_right._data
            )
        ],
        dtype,
    )
