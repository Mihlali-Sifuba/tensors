"""Python-native axis permutation."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)

if TYPE_CHECKING:
    from tensors.dtype import DataType


def transpose(
    value: Any,
    permutation: tuple[int, ...],
    *,
    input_shape: tuple[int, ...],
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> PythonStorage:
    """Copy compact values into the requested axis order."""
    if len(input_shape) == 2 and permutation == (1, 0):
        rows, columns = input_shape
        return PythonStorage.from_values(
            (
                value[row * columns + column]
                for column in range(columns)
                for row in range(rows)
            ),
            dtype,
        )
    inverse = [0] * len(input_shape)
    for output_axis, input_axis in enumerate(permutation):
        inverse[input_axis] = output_axis
    result = []
    size = 1
    for dimension in output_shape:
        size *= dimension
    for index in range(size):
        output_coordinates = linear_index_to_coordinates(index, output_shape)
        input_coordinates = tuple(
            output_coordinates[inverse[input_axis]]
            for input_axis in range(len(input_shape))
        )
        result.append(
            value[coordinates_to_linear_index(input_coordinates, input_shape)]
        )
    return PythonStorage.from_values(result, dtype)
