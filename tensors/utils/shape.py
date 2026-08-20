"""Pure helpers for tensor shapes and row-major indexing."""

from typing import Tuple


def _validate_shape_dimensions(shape: Tuple[int, ...]) -> None:
    """Reject shapes containing invalid dimensions."""
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError(
                f"Invalid shape {shape}: dimensions must be non-negative integers"
            )


def shape_size(shape: Tuple[int, ...]) -> int:
    """Return the number of elements described by ``shape``."""
    _validate_shape_dimensions(shape)
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def row_major_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return row-major strides for ``shape``."""
    _validate_shape_dimensions(shape)
    stride = 1
    strides = []
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    return tuple(reversed(strides))


def index_to_coordinates(index: int, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert a valid row-major flat index to coordinates for ``shape``."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")

    size = shape_size(shape)
    if not 0 <= index < size:
        raise IndexError(f"Flat index {index} is out of range for shape {shape}")

    coordinates = []
    for stride in row_major_strides(shape):
        coordinate = index // stride
        coordinates.append(coordinate)
        index %= stride
    return tuple(coordinates)


def coordinates_to_index(
    coordinates: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> int:
    """Convert valid row-major coordinates to a flat index."""
    _validate_shape_dimensions(shape)
    if len(coordinates) != len(shape):
        raise ValueError(
            f"Coordinate rank {len(coordinates)} does not match shape rank {len(shape)}"
        )

    for coordinate, dimension in zip(coordinates, shape):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise TypeError("coordinates must contain only integers")
        if not 0 <= coordinate < dimension:
            raise IndexError(
                f"Coordinate {coordinates} is out of range for shape {shape}"
            )

    return sum(
        coordinate * stride
        for coordinate, stride in zip(coordinates, row_major_strides(shape))
    )


__all__ = [
    "shape_size",
    "row_major_strides",
    "index_to_coordinates",
    "coordinates_to_index",
]
