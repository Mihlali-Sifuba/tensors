"""Coordinate validation and logical-to-physical address mapping."""

from typing import Tuple

from ..shape import Shape
from ..strides import Strides


def coordinates_to_storage_index(
    coordinates: Tuple[int, ...],
    shape: Shape,
    strides: Strides,
    offset: int = 0,
) -> int:
    """Map valid logical coordinates to a physical storage position."""
    if len(strides) != shape.rank:
        raise ValueError(
            f"Stride rank {len(strides)} does not match shape rank {shape.rank}"
        )
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if len(coordinates) != shape.rank:
        raise ValueError(
            f"Coordinate rank {len(coordinates)} does not match "
            f"shape rank {shape.rank}"
        )

    for coordinate, dimension in zip(coordinates, shape):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise TypeError("coordinates must contain only integers")
        if not 0 <= coordinate < dimension:
            raise IndexError(
                f"Coordinate {coordinates} is out of range for shape {shape}"
            )

    return offset + sum(
        coordinate * stride
        for coordinate, stride in zip(coordinates, strides)
    )


def tensor_indices_to_storage_index(
    indices: Tuple[int, ...],
    shape: Tuple[int, ...] | Shape,
    strides: Tuple[int, ...] | Strides | None = None,
    offset: int = 0,
) -> int:
    """Normalize tensor indices and return their physical storage index."""
    normalized_shape = (
        shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
    )
    normalized_strides = (
        Strides.contiguous(normalized_shape)
        if strides is None
        else (
            strides
            if isinstance(strides, Strides)
            else Strides.from_iterable(strides)
        )
    )
    if len(indices) != normalized_shape.rank:
        raise IndexError(
            f"Expected {normalized_shape.rank} indices, got {len(indices)}"
        )

    normalized_indices = []
    for index, dimension_size in zip(indices, normalized_shape):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Tensor indices must be integers, not bools")
        normalized_index = index + dimension_size if index < 0 else index
        if not 0 <= normalized_index < dimension_size:
            raise IndexError("Index out of range")
        normalized_indices.append(normalized_index)

    return coordinates_to_storage_index(
        tuple(normalized_indices),
        normalized_shape,
        normalized_strides,
        offset,
    )


__all__ = [
    "coordinates_to_storage_index",
    "tensor_indices_to_storage_index",
]
