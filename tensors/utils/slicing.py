"""Utilities for normalizing tensor slice selections."""

from itertools import product
from typing import List, Tuple, Union

from ..shape import Shape
from ..strides import Strides
from .coordinates import coordinates_to_linear_index
from .indexing import coordinates_to_storage_index


def slice_ranges_and_shape_from_key(
    key: Tuple[Union[int, slice], ...],
    shape: Shape | Tuple[int, ...],
) -> Tuple[List[range], Shape]:
    """Normalize a slice key into dimension ranges and its result shape."""
    normalized_shape = (
        shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
    )
    ndim = len(normalized_shape)
    if len(key) > ndim:
        raise IndexError(f"Too many indices: {len(key)} for {ndim}D tensor")

    ranges = []
    new_shape = []
    for dimension, index in enumerate(key):
        if isinstance(index, bool):
            raise TypeError("Boolean tensor indices are not supported")
        if isinstance(index, int):
            normalized_index = (
                index if index >= 0 else index + normalized_shape[dimension]
            )
            if not 0 <= normalized_index < normalized_shape[dimension]:
                raise IndexError("Index out of range")
            ranges.append(range(normalized_index, normalized_index + 1))
        elif isinstance(index, slice):
            dimension_range = range(*index.indices(normalized_shape[dimension]))
            ranges.append(dimension_range)
            new_shape.append(len(dimension_range))
        else:
            raise TypeError(f"Unsupported index type in tuple: {type(index)}")

    while len(ranges) < ndim:
        dimension = len(ranges)
        dimension_range = range(normalized_shape[dimension])
        ranges.append(dimension_range)
        new_shape.append(normalized_shape[dimension])

    return ranges, Shape(*new_shape)


def storage_indices_from_ranges(
    ranges: List[range],
    shape: Shape | Tuple[int, ...],
    strides: Tuple[int, ...] | Strides,
    offset: int = 0,
) -> List[int]:
    """Return physical storage indices selected by dimension ranges."""
    normalized_shape = (
        shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
    )
    normalized_strides = (
        strides
        if isinstance(strides, Strides)
        else Strides.from_iterable(strides)
    )
    return [
        coordinates_to_storage_index(
            coordinates,
            normalized_shape,
            normalized_strides,
            offset,
        )
        for coordinates in product(*ranges)
    ]


def logical_linear_indices_from_ranges(
    ranges: List[range],
    shape: Shape | Tuple[int, ...],
) -> List[int]:
    """Return logical linear indices selected by dimension ranges."""
    normalized_shape = (
        shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
    )
    return [
        coordinates_to_linear_index(coordinates, normalized_shape)
        for coordinates in product(*ranges)
    ]


__all__ = [
    "logical_linear_indices_from_ranges",
    "slice_ranges_and_shape_from_key",
    "storage_indices_from_ranges",
]
