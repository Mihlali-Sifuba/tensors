"""Python-native slice scattering."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.shape import Shape

if TYPE_CHECKING:
    from tensors.dtype import DataType


def slice_scatter(
    value: Any,
    indices: list[int],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> PythonStorage:
    """Scatter compact source values into a new zero-filled buffer."""
    result = [0.0] * Shape.from_iterable(output_shape).size
    for logical_linear_index, grad_value in zip(indices, value):
        result[logical_linear_index] += grad_value
    return PythonStorage.from_values(result, dtype)
