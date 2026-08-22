"""Utilities for normalizing tensor slice selections."""

from typing import List, Tuple, Union


def slice_ranges_and_shape_from_key(
    key: Tuple[Union[int, slice], ...],
    shape: Tuple[int, ...],
) -> Tuple[List[range], Tuple[int, ...]]:
    """Normalize a slice key into dimension ranges and its result shape."""
    ndim = len(shape)
    if len(key) > ndim:
        raise IndexError(f"Too many indices: {len(key)} for {ndim}D tensor")

    ranges = []
    new_shape = []
    for dimension, index in enumerate(key):
        if isinstance(index, bool):
            raise TypeError("Boolean tensor indices are not supported")
        if isinstance(index, int):
            normalized_index = index if index >= 0 else index + shape[dimension]
            if not 0 <= normalized_index < shape[dimension]:
                raise IndexError("Index out of range")
            ranges.append(range(normalized_index, normalized_index + 1))
        elif isinstance(index, slice):
            dimension_range = range(*index.indices(shape[dimension]))
            ranges.append(dimension_range)
            new_shape.append(len(dimension_range))
        else:
            raise TypeError(f"Unsupported index type in tuple: {type(index)}")

    while len(ranges) < ndim:
        dimension = len(ranges)
        dimension_range = range(shape[dimension])
        ranges.append(dimension_range)
        new_shape.append(shape[dimension])

    return ranges, tuple(new_shape)


__all__ = ["slice_ranges_and_shape_from_key"]
