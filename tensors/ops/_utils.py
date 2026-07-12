"""Shared shape helpers for differentiable algebra operations."""

from __future__ import annotations

from ..tensor import Tensor, _coordinates, _flat_index, _shape_size


def unbroadcast(gradient: Tensor, shape: tuple[int, ...]) -> Tensor:
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
    values = [0.0] * _shape_size(shape)
    padding = gradient.ndim - len(shape)
    for index, value in enumerate(gradient._data):
        gradient_coordinates = _coordinates(index, gradient.shape)
        source_coordinates = tuple(
            0 if source_dimension == 1 else coordinate
            for source_dimension, coordinate in zip(padded_shape, gradient_coordinates)
        )[padding:]
        values[_flat_index(source_coordinates, shape)] += value

    return Tensor(values, dtype=gradient.dtype, shape=shape)


__all__ = ["unbroadcast"]
