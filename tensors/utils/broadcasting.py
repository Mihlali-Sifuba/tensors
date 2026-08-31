"""Helpers for NumPy-style tensor broadcasting."""

from collections.abc import Iterable
from typing import Tuple

from ..shape import Shape
from ..tensor import Tensor
from .coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


def broadcast_to(tensor: Tensor, shape: Shape | Iterable[int]) -> Tensor:
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
        output_coordinates = linear_index_to_coordinates(
            output_index,
            output_shape,
        )
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, output_coordinates)
        )[padding:]
        values.append(
            tensor._data[
                coordinates_to_linear_index(source_coordinates, tensor.shape)
            ]
        )

    return Tensor(values, dtype=tensor.dtype, shape=output_shape)


def broadcast_tensors(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = a.shape.broadcast_with(b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


__all__ = ["broadcast_to", "broadcast_tensors"]
