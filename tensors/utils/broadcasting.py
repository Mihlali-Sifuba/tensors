"""NumPy-style tensor broadcasting.

Broadcasting is one responsibility: given a tensor and a target shape, return
a tensor whose logical values and shape are that tensor broadcast to it.
Nothing here performs arithmetic, comparison or selection, resolves a result
dtype, or selects a backend — an operation broadcasts its operands and then
computes for itself, so this module never needs to know which operation
consumes its result.

:meth:`Shape.broadcast_with` agrees the common shape between two operands;
:func:`broadcast_to` expands one operand to it; :func:`broadcast_tensors` does
both for a pair. :func:`broadcast_source_indices` states the expansion as the
mapping it is, for a caller that applies it to values of its own.
"""

from collections.abc import Iterable
from itertools import product
from typing import Tuple

from ..shape import Shape
from ..strides import Strides
from ..tensor import Tensor


def broadcast_source_indices(
    source_shape: Shape | Iterable[int], target_shape: Shape | Iterable[int]
) -> list[int]:
    """Map each position of the broadcast result to the position it reads.

    Broadcasting repeats values rather than computing them, so the whole of
    it is this mapping: every position of the target shape names the one
    logical position of the source that supplies it. A stretched axis is
    simply several target positions naming the same source position.

    Returning the mapping instead of the values lets a caller that already
    holds the source apply it without reading those values on the host. The
    indices are logical row-major positions in both spaces and are metadata,
    so this belongs to no backend and moves nothing.

    This is internal. :func:`broadcast_to` materializes the mapping into a
    tensor, which is what a caller computing from the result wants.
    """
    source = Shape.from_iterable(source_shape)
    target = Shape.from_iterable(target_shape)
    if source.rank > target.rank:
        raise ValueError(f"Shape {source} cannot be broadcast to {target}")

    padding = target.rank - source.rank
    padded_shape = (1,) * padding + tuple(source)
    for source_dimension, target_dimension in zip(padded_shape, target):
        if source_dimension not in {1, target_dimension}:
            raise ValueError(f"Shape {source} cannot be broadcast to {target}")

    source_strides = (0,) * padding + tuple(Strides.contiguous(source))
    broadcast_strides = tuple(
        0 if source_dimension == 1 else stride
        for source_dimension, stride in zip(padded_shape, source_strides)
    )
    return [
        sum(
            coordinate * stride
            for coordinate, stride in zip(coordinates, broadcast_strides)
        )
        for coordinates in product(*(range(dimension) for dimension in target))
    ]


def broadcast_to(tensor: Tensor, shape: Shape | Iterable[int]) -> Tensor:
    """Materialize ``tensor`` at ``shape`` using singleton dimensions."""
    output_shape = Shape.from_iterable(shape)
    if tensor.shape == output_shape:
        return tensor
    indices = broadcast_source_indices(tensor.shape, output_shape)
    source_data = tensor._data
    return Tensor._from_values(
        [source_data[index] for index in indices], tensor.dtype, output_shape
    )


def broadcast_tensors(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Broadcast two tensors to a shared NumPy-style shape."""
    shape = a.shape.broadcast_with(b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


__all__ = ["broadcast_source_indices", "broadcast_to", "broadcast_tensors"]
