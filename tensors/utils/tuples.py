"""Utilities for validating and transforming tuples."""

from typing import Tuple

from .shape import coordinates_to_index


def indices_to_flat_index(
    indices: Tuple[int, ...],
    shape: Tuple[int, ...],
) -> int:
    """Normalize tensor indices and return their row-major flat index."""
    if len(indices) != len(shape):
        raise IndexError(
            f"Expected {len(shape)} indices, got {len(indices)}"
        )

    normalized_indices = []
    for index, dimension_size in zip(indices, shape):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Tensor indices must be integers, not bools")
        normalized_index = index + dimension_size if index < 0 else index
        if not 0 <= normalized_index < dimension_size:
            raise IndexError("Index out of range")
        normalized_indices.append(normalized_index)

    return coordinates_to_index(tuple(normalized_indices), shape)


__all__ = ["indices_to_flat_index"]
