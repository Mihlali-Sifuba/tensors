"""Helpers for NumPy-style tensor broadcasting."""

from typing import Tuple

from ..shape import Shape
from ..tensor import Tensor
from .shape import coordinates_to_index, index_to_coordinates


def broadcast_shape(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
) -> Shape:
    """Return the NumPy-style broadcast shape for two tensor shapes."""
    normalized_a = Shape.from_iterable(a_shape)
    normalized_b = Shape.from_iterable(b_shape)

    dimensions = []
    for a_dimension, b_dimension in zip(
        reversed(normalized_a),
        reversed(normalized_b),
    ):
        if a_dimension == b_dimension:
            dimensions.append(a_dimension)
        elif a_dimension == 1:
            dimensions.append(b_dimension)
        elif b_dimension == 1:
            dimensions.append(a_dimension)
        else:
            raise ValueError(
                f"Shapes {normalized_a} and {normalized_b} cannot be broadcast"
            )

    longer_shape = (
        normalized_a
        if normalized_a.rank > normalized_b.rank
        else normalized_b
    )
    matched_dimensions = min(normalized_a.rank, normalized_b.rank)
    unmatched_dimensions = len(longer_shape) - matched_dimensions
    dimensions.extend(reversed(longer_shape[:unmatched_dimensions]))
    return Shape(*reversed(dimensions))


def broadcast_to(tensor: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Materialize ``tensor`` at ``shape`` using singleton dimensions."""
    output_shape = Shape.from_iterable(shape)
    output_size = output_shape.size
    if tensor.shape == output_shape:
        return tensor
    if tensor.shape.rank > output_shape.rank:
        raise ValueError(
            f"Shape {tensor.shape} cannot be broadcast to {output_shape}"
        )

    padded_shape = (
        (1,) * (output_shape.rank - tensor.ndim) + tensor.shape
    )
    for source_dimension, target_dimension in zip(padded_shape, output_shape):
        if source_dimension not in {1, target_dimension}:
            raise ValueError(
                f"Shape {tensor.shape} cannot be broadcast to {output_shape}"
            )

    values = []
    padding = output_shape.rank - tensor.ndim
    for output_index in range(output_size):
        output_coordinates = index_to_coordinates(output_index, output_shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, output_coordinates)
        )[padding:]
        values.append(
            tensor._data[coordinates_to_index(source_coordinates, tensor.shape)]
        )

    return Tensor(values, dtype=tensor.dtype, shape=output_shape)


def broadcast_tensors(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = broadcast_shape(a.shape, b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


__all__ = ["broadcast_shape", "broadcast_to", "broadcast_tensors"]
