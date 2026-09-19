"""Reference multiplication for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def multiply(
    left: Tensor | int | float,
    right: Tensor | int | float,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Multiply each broadcast pair with Python scalar semantics."""
    from tensors.tensor import Tensor
    from tensors.utils.broadcasting import broadcast_binary_values

    def evaluate(x, y):
        return x * y

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        values = broadcast_binary_values(left, right, output_shape, evaluate)
    elif isinstance(left, Tensor):
        values = [evaluate(x, right) for x in left._data]
    elif isinstance(right, Tensor):
        values = [evaluate(left, y) for y in right._data]
    else:
        values = [evaluate(left, right)]
    return PythonStorage.from_arithmetic(values, dtype)
