"""Helpers for NumPy-style tensor broadcasting."""

from typing import Tuple

from ..tensor import Tensor
from .shape import coordinates_to_index, index_to_coordinates, shape_size


def broadcast_shape(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Return the NumPy-style broadcast shape for two tensor shapes."""
    shape_size(a_shape)
    shape_size(b_shape)

    dimensions = []
    for a_dimension, b_dimension in zip(reversed(a_shape), reversed(b_shape)):
        if a_dimension == b_dimension:
            dimensions.append(a_dimension)
        elif a_dimension == 1:
            dimensions.append(b_dimension)
        elif b_dimension == 1:
            dimensions.append(a_dimension)
        else:
            raise ValueError(f"Shapes {a_shape} and {b_shape} cannot be broadcast")

    longer_shape = a_shape if len(a_shape) > len(b_shape) else b_shape
    matched_dimensions = min(len(a_shape), len(b_shape))
    unmatched_dimensions = len(longer_shape) - matched_dimensions
    dimensions.extend(reversed(longer_shape[:unmatched_dimensions]))
    return tuple(reversed(dimensions))


def broadcast_to(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Materialize ``tensor`` at ``shape`` using singleton dimensions."""
    output_size = shape_size(shape)
    if tensor.shape == shape:
        return tensor
    if len(tensor.shape) > len(shape):
        raise ValueError(f"Shape {tensor.shape} cannot be broadcast to {shape}")

    padded_shape = (1,) * (len(shape) - tensor.ndim) + tensor.shape
    for source_dimension, target_dimension in zip(padded_shape, shape):
        if source_dimension not in {1, target_dimension}:
            raise ValueError(f"Shape {tensor.shape} cannot be broadcast to {shape}")

    values = []
    padding = len(shape) - tensor.ndim
    for output_index in range(output_size):
        output_coordinates = index_to_coordinates(output_index, shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, output_coordinates)
        )[padding:]
        values.append(
            tensor._data[coordinates_to_index(source_coordinates, tensor.shape)]
        )

    return Tensor(values, dtype=tensor.dtype, shape=shape)


def broadcast_tensors(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = broadcast_shape(a.shape, b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


__all__ = ["broadcast_shape", "broadcast_to", "broadcast_tensors"]
