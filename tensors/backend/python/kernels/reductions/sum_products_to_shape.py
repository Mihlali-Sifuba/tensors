"""Reference fused product summation to a broadcast shape, in Python."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.shape import Shape
from tensors.tensor import Tensor
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


def sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage | None:
    """Sum the products of two broadcast operands down to one shape."""
    from tensors.utils.summation import stable_product_sum
    from tensors.utils.broadcasting import broadcast_tensors

    expanded_gradient, expanded_factor = broadcast_tensors(gradient, factor)
    if len(shape) > expanded_gradient.ndim:
        raise ValueError(
            f"Cannot reduce gradient shape {expanded_gradient.shape} to {shape}"
        )
    target = Shape.from_iterable(shape)
    if expanded_gradient.shape == target:
        values = [
            stable_product_sum([(float(left), float(right))])
            for left, right in zip(expanded_gradient._data, expanded_factor._data)
        ]
        return Tensor._from_values(values, gradient.dtype, target)._storage
    if target.size == 1:
        values = [
            stable_product_sum(
                [
                    (float(left), float(right))
                    for left, right in zip(
                        expanded_gradient._data, expanded_factor._data
                    )
                ]
            )
        ]
        return Tensor._from_values(values, gradient.dtype, target)._storage
    padded_shape = (1,) * (expanded_gradient.ndim - len(shape)) + shape
    padding = expanded_gradient.ndim - len(shape)
    groups: list[list[tuple[float, float]]] = [
        [] for _ in range(Shape.from_iterable(shape).size)
    ]
    for index, (left, right) in enumerate(
        zip(expanded_gradient._data, expanded_factor._data)
    ):
        coordinates = linear_index_to_coordinates(index, expanded_gradient.shape)
        source_coordinates = tuple(
            (
                0 if source_dimension == 1 else coordinate
                for source_dimension, coordinate in zip(padded_shape, coordinates)
            )
        )[padding:]
        source_index = coordinates_to_linear_index(source_coordinates, shape)
        groups[source_index].append((float(left), float(right)))
    values = [stable_product_sum(group) for group in groups]
    return PythonStorage.from_values(values, gradient.dtype)
