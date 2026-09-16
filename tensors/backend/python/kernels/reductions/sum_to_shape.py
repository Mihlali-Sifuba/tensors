"""Reference summation to a broadcast shape, in Python."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape
from tensors.tensor import Tensor
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


def sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage | None:
    """Sum broadcast contributions back down to the original shape."""
    target = Shape.from_iterable(shape)
    if target.size == 1:
        if gradient.dtype.kind == "floating":
            from tensors.math.sum import _stable_float_sum

            total = _stable_float_sum([float(value) for value in gradient._data])
        else:
            total = sum(gradient._data)
        return Tensor._from_values([total], gradient.dtype, target)._storage
    padded_shape = (1,) * (gradient.ndim - len(shape)) + shape
    groups: list[list[int | float]] = [[] for _ in range(target.size)]
    padding = gradient.ndim - len(shape)
    for index, value in enumerate(gradient._data):
        gradient_coordinates = linear_index_to_coordinates(index, gradient.shape)
        source_coordinates = tuple(
            (
                0 if source_dimension == 1 else coordinate
                for source_dimension, coordinate in zip(
                    padded_shape, gradient_coordinates
                )
            )
        )[padding:]
        groups[coordinates_to_linear_index(source_coordinates, shape)].append(value)
    if gradient.dtype.kind == "floating":
        from tensors.math.sum import _stable_float_sum

        values = [
            _stable_float_sum([float(value) for value in group]) for group in groups
        ]
    else:
        values = [sum(group) for group in groups]
    return PythonStorage.from_values(values, gradient.dtype)
