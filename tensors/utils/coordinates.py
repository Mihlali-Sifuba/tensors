"""Canonical row-major logical coordinate conversion helpers."""

from typing import Tuple

from ..shape import Shape
from ..strides import Strides
from .indexing import coordinates_to_storage_index


def linear_index_to_coordinates(
    index: int,
    shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Convert a valid logical row-major linear index to coordinates."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")

    normalized_shape = Shape.from_iterable(shape)
    size = normalized_shape.size
    if not 0 <= index < size:
        raise IndexError(
            f"Logical linear index {index} is out of range for shape {shape}"
        )

    coordinates = []
    for stride in Strides.contiguous(normalized_shape):
        coordinate = index // stride
        coordinates.append(coordinate)
        index %= stride
    return tuple(coordinates)


def coordinates_to_linear_index(
    coordinates: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> int:
    """Convert valid coordinates to a logical row-major linear index."""
    normalized_shape = Shape.from_iterable(shape)
    return coordinates_to_storage_index(
        coordinates,
        normalized_shape,
        Strides.contiguous(normalized_shape),
    )


__all__ = [
    "coordinates_to_linear_index",
    "linear_index_to_coordinates",
]
