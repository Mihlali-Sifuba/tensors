"""Reference addition for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def add(
    left: Tensor | int | float,
    right: Tensor | int | float,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Add each broadcast pair with Python scalar semantics."""
    from tensors.tensor import Tensor
    from tensors.utils.broadcasting import broadcast_to

    def evaluate(x, y):
        return x + y

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        # Broadcasting first, then the operation: each is one
        # responsibility, and neither needs to know the other.
        left_values = broadcast_to(left, output_shape)._data
        right_values = broadcast_to(right, output_shape)._data
        values = [evaluate(x, y) for x, y in zip(left_values, right_values)]
    elif isinstance(left, Tensor):
        values = [evaluate(x, right) for x in left._data]
    elif isinstance(right, Tensor):
        values = [evaluate(left, y) for y in right._data]
    else:
        values = [evaluate(left, right)]
    return PythonStorage.from_arithmetic(values, dtype)
