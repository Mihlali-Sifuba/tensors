"""Reference axis permutation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def transpose(
    value: Tensor, permutation: tuple[int, ...], *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Copy the values into the requested axis order."""
    tensor = value
    shape = output_shape
    if tensor.ndim == 2 and permutation == (1, 0):
        rows, columns = tensor.shape
        source = tensor._data
        storage = PythonStorage.from_values(
            (
                source[row * columns + column]
                for column in range(columns)
                for row in range(rows)
            ),
            tensor.dtype,
        )
        return storage
    inverse = [0] * tensor.ndim
    for output_axis, input_axis in enumerate(permutation):
        inverse[input_axis] = output_axis
    values = []
    for index in range(tensor.size):
        output_coordinates = linear_index_to_coordinates(index, shape)
        input_coordinates = tuple(
            (
                output_coordinates[inverse[input_axis]]
                for input_axis in range(tensor.ndim)
            )
        )
        values.append(
            tensor._data[coordinates_to_linear_index(input_coordinates, tensor.shape)]
        )
    return PythonStorage.from_values(values, tensor.dtype)
