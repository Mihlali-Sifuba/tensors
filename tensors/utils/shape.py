"""Pure helpers for tensor shapes and row-major indexing."""

from typing import Iterable, Tuple


def normalize_shape(shape: Iterable[int]) -> Tuple[int, ...]:
    """Return ``shape`` as a tuple after validating its dimensions."""
    try:
        normalized_shape = tuple(shape)
    except TypeError as exc:
        raise TypeError(
            "shape must be an iterable of non-negative integers"
        ) from exc

    for dimension in normalized_shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
        ):
            raise ValueError(
                f"Invalid shape {normalized_shape}: "
                "dimensions must be non-negative integers"
            )
    return normalized_shape


def shape_size(shape: Tuple[int, ...]) -> int:
    """Return the number of elements described by ``shape``."""
    normalized_shape = normalize_shape(shape)
    size = 1
    for dimension in normalized_shape:
        size *= dimension
    return size


def row_major_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return row-major strides for ``shape``."""
    normalized_shape = normalize_shape(shape)
    stride = 1
    strides = []
    for dimension in reversed(normalized_shape):
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
    normalized_shape = normalize_shape(shape)
    if len(coordinates) != len(normalized_shape):
        raise ValueError(
            f"Coordinate rank {len(coordinates)} does not match "
            f"shape rank {len(normalized_shape)}"
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
            row_major_strides(normalized_shape),
        )
    )


__all__ = [
    "normalize_shape",
    "shape_size",
    "row_major_strides",
    "index_to_coordinates",
    "coordinates_to_index",
]
