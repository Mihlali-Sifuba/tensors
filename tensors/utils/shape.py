"""Canonical row-major coordinate conversion helpers."""

from typing import Tuple

from ..shape import Shape
from ..strides import Strides
from .indexing import coordinates_to_storage_index


def index_to_coordinates(index: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert a valid row-major flat index to coordinates for ``shape``."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")

    normalized_shape = Shape.from_iterable(shape)
    size = normalized_shape.size
    if not 0 <= index < size:
        raise IndexError(f"Flat index {index} is out of range for shape {shape}")

    coordinates = []
    for stride in Strides.contiguous(normalized_shape):
        coordinate = index // stride
        coordinates.append(coordinate)
        index %= stride
    return tuple(coordinates)


def coordinates_to_index(
    coordinates: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> int:
    """Convert valid row-major coordinates to a flat index."""
    normalized_shape = Shape.from_iterable(shape)
    return coordinates_to_storage_index(
        coordinates,
        normalized_shape,
        Strides.contiguous(normalized_shape),
    )


__all__ = [
    "index_to_coordinates",
    "coordinates_to_index",
]
