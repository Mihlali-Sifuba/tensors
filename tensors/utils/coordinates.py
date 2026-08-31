"""Canonical row-major logical coordinate conversion helpers."""

from typing import Tuple

from ..shape import Shape
from ..strides import Strides


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
    if len(coordinates) != normalized_shape.rank:
        raise ValueError(
            f"Coordinate rank {len(coordinates)} does not match "
            f"shape rank {normalized_shape.rank}"
        )

    for coordinate, dimension in zip(coordinates, normalized_shape):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise TypeError("coordinates must contain only integers")
        if not 0 <= coordinate < dimension:
            raise IndexError(
                f"Coordinate {coordinates} is out of range for shape "
                f"{normalized_shape}"
            )

    return sum(
        coordinate * stride
        for coordinate, stride in zip(
            coordinates,
            Strides.contiguous(normalized_shape),
        )
    )


__all__ = [
    "coordinates_to_linear_index",
    "linear_index_to_coordinates",
]
