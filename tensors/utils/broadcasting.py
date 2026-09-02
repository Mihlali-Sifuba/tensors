"""Helpers for NumPy-style tensor broadcasting."""

from collections.abc import Callable, Iterable
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


def broadcast_binary_values(
    left: Tensor,
    right: Tensor,
    shape: Shape | Iterable[int],
    operation: Callable[[object, object], object],
) -> list[object]:
    """Apply a binary operation while walking broadcast offsets once."""
    output_shape = Shape.from_iterable(shape)
    if left.shape.broadcast_with(right.shape) != output_shape:
        raise ValueError(
            f"Shapes {left.shape} and {right.shape} do not broadcast to "
            f"{output_shape}"
        )
    left_data = left._data
    right_data = right._data
    if left.shape == right.shape:
        return [operation(a, b) for a, b in zip(left_data, right_data)]
    if left.size == 1:
        scalar = left_data[0]
        return [operation(scalar, value) for value in right_data]
    if right.size == 1:
        scalar = right_data[0]
        return [operation(value, scalar) for value in left_data]

    rank = output_shape.rank
    left_padding = rank - left.ndim
    right_padding = rank - right.ndim
    left_shape = (1,) * left_padding + tuple(left.shape)
    right_shape = (1,) * right_padding + tuple(right.shape)
    left_physical = (0,) * left_padding + tuple(Strides.contiguous(left.shape))
    right_physical = (0,) * right_padding + tuple(Strides.contiguous(right.shape))
    left_strides = tuple(
        0 if dimension == 1 else stride
        for dimension, stride in zip(left_shape, left_physical)
    )
    right_strides = tuple(
        0 if dimension == 1 else stride
        for dimension, stride in zip(right_shape, right_physical)
    )

    coordinates = [0] * rank
    left_offset = 0
    right_offset = 0
    values: list[object] = []
    append = values.append
    for _ in range(output_shape.size):
        append(operation(left_data[left_offset], right_data[right_offset]))
        for axis in range(rank - 1, -1, -1):
            coordinates[axis] += 1
            left_offset += left_strides[axis]
            right_offset += right_strides[axis]
            if coordinates[axis] < output_shape[axis]:
                break
            coordinates[axis] = 0
            left_offset -= left_strides[axis] * output_shape[axis]
            right_offset -= right_strides[axis] * output_shape[axis]
    return values


__all__ = ["broadcast_binary_values", "broadcast_to", "broadcast_tensors"]
