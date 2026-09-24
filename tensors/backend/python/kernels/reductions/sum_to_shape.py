"""Reference summation to a broadcast shape, in Python."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from tensors.shape import Shape
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


def sum_to_shape(
    gradient_values,
    input_shape: tuple[int, ...],
    shape: tuple[int, ...],
    *,
    dtype: DataType,
) -> Storage | None:
    """Sum broadcast contributions back down to the original shape."""
    target = Shape.from_iterable(shape)
    if target.size == 1:
        if dtype.kind == "floating":
            from tensors.utils.summation import stable_float_sum

            total = stable_float_sum([float(value) for value in gradient_values])
        else:
            total = sum(gradient_values)
        return PythonStorage.from_values([total], dtype)
    padded_shape = (1,) * (len(input_shape) - len(shape)) + shape
    groups: list[list[int | float]] = [[] for _ in range(target.size)]
    padding = len(input_shape) - len(shape)
    for index, value in enumerate(gradient_values):
        gradient_coordinates = linear_index_to_coordinates(index, input_shape)
        source_coordinates = tuple(
            (
                0 if source_dimension == 1 else coordinate
                for source_dimension, coordinate in zip(
                    padded_shape, gradient_coordinates
                )
            )
        )[padding:]
        groups[coordinates_to_linear_index(source_coordinates, shape)].append(value)
    if dtype.kind == "floating":
        from tensors.utils.summation import stable_float_sum

        values = [
            stable_float_sum([float(value) for value in group]) for group in groups
        ]
    else:
        values = [sum(group) for group in groups]
    return PythonStorage.from_values(values, dtype)
