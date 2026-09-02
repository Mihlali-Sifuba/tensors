"""Helpers for NumPy-style tensor broadcasting."""

from collections.abc import Iterable
from itertools import product
from typing import Tuple

from ..shape import Shape
from ..strides import Strides
from ..tensor import Tensor


def broadcast_to(tensor: Tensor, shape: Shape | Iterable[int]) -> Tensor:
    """Materialize ``tensor`` at ``shape`` using singleton dimensions."""
    output_shape = Shape.from_iterable(shape)
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

    padding = output_shape.rank - tensor.ndim
    source_data = tensor._data
    source_strides = (0,) * padding + tuple(Strides.contiguous(tensor.shape))
    broadcast_strides = tuple(
        0 if source_dimension == 1 else stride
        for source_dimension, stride in zip(padded_shape, source_strides)
    )
    values = [
        source_data[sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, broadcast_strides)
        )]
        for coordinates in product(*(range(dimension) for dimension in output_shape))
    ]

    return Tensor(values, dtype=tensor.dtype, shape=output_shape)


def broadcast_tensors(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = a.shape.broadcast_with(b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


__all__ = ["broadcast_to", "broadcast_tensors"]
