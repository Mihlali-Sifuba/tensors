"""Shared gradient-shape helpers for differentiable operations."""

from __future__ import annotations

from ..tensor import Tensor
from ..utils.shape import coordinates_to_index, index_to_coordinates, shape_size


def sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reduce a broadcasted ``gradient`` back to an input ``shape``.

    During a forward broadcast, a value can participate in several output
    positions. Reverse-mode differentiation must sum those contributions into
    the corresponding original position.
    """
    if gradient.shape == shape:
        return gradient
    if len(shape) > gradient.ndim:
        raise ValueError(f"Cannot reduce gradient shape {gradient.shape} to {shape}")

    padded_shape = (1,) * (gradient.ndim - len(shape)) + shape
    values = [0.0] * shape_size(shape)
    padding = gradient.ndim - len(shape)
    for index, value in enumerate(gradient._data):
        gradient_coordinates = index_to_coordinates(index, gradient.shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, gradient_coordinates)
        )[padding:]
        values[coordinates_to_index(source_coordinates, shape)] += value

    return Tensor(values, dtype=gradient.dtype, shape=shape)


def sum_to_shape_graph(gradient, shape: tuple[int, ...]):
    """Differentiably reduce a broadcasted Variable back to ``shape``."""
    from ..math import reshape, sum

    if gradient.shape == shape:
        return gradient
    if len(shape) > gradient.ndim:
        raise ValueError(f"Cannot reduce gradient shape {gradient.shape} to {shape}")

    padding = gradient.ndim - len(shape)
    padded_shape = (1,) * padding + shape
    axes = tuple(
        axis
        for axis, (source, target) in enumerate(zip(gradient.shape, padded_shape))
        if target == 1 and source != 1
    )
    reduced = sum(gradient, axis=axes, keepdims=True) if axes else gradient
    return reshape(reduced, shape) if reduced.shape != shape else reduced


__all__ = ["sum_to_shape", "sum_to_shape_graph"]
