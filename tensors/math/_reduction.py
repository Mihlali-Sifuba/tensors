"""Shared shape and indexing helpers for axis-aware reductions."""

from typing import Optional, TypeAlias

from ..shape import Shape
from ..tensor import Tensor
from ..utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)


Axis: TypeAlias = Optional[int | tuple[int, ...] | list[int]]


def immutable_axis(axis: Axis) -> int | tuple[int, ...] | None:
    """Copy a mutable axis list into immutable graph metadata."""
    return tuple(axis) if isinstance(axis, list) else axis


def normalize_axes(ndim: int, axis: Axis) -> tuple[int, ...]:
    """Return validated, non-negative reduction axes in ascending order."""
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, bool):
        raise TypeError("axis must be an integer, tuple of integers, or None")
    if isinstance(axis, int):
        requested = (axis,)
    elif isinstance(axis, (tuple, list)):
        requested = tuple(axis)
    else:
        raise TypeError("axis must be an integer, tuple of integers, or None")

    normalized = []
    for value in requested:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("axis entries must be integers")
        original = value
        if value < 0:
            value += ndim
        if not 0 <= value < ndim:
            raise ValueError(f"Axis {original} out of bounds for {ndim}D tensor")
        if value in normalized:
            raise ValueError(f"Duplicate axis {original}")
        normalized.append(value)
    return tuple(sorted(normalized))


def reduction_shape(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    keepdims: bool,
) -> tuple[int, ...]:
    """Return the output shape produced by reducing ``axes``."""
    if not isinstance(keepdims, bool):
        raise TypeError("keepdims must be a bool")
    if keepdims:
        return tuple(1 if index in axes else size for index, size in enumerate(shape))
    return tuple(size for index, size in enumerate(shape) if index not in axes)


def reduction_size(shape: tuple[int, ...], axes: tuple[int, ...]) -> int:
    """Return the number of input values contributing to each output value."""
    return Shape(*(shape[axis] for axis in axes)).size


def reduction_groups(
    value: Tensor,
    axis: Axis,
    keepdims: bool,
    *,
    scalar_as_vector: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], list[list[int]]]:
    """Group flat input indices by their corresponding reduction output."""
    if not isinstance(keepdims, bool):
        raise TypeError("keepdims must be a bool")
    axes = normalize_axes(value.ndim, axis)
    output_shape = reduction_shape(value.shape, axes, keepdims)
    scalar_output_as_vector = scalar_as_vector and axis is None and not keepdims
    if scalar_output_as_vector:
        output_shape = (1,)
    groups = [[] for _ in range(Shape.from_iterable(output_shape).size)]
    axes_set = set(axes)

    for input_index in range(value.size):
        input_coordinates = linear_index_to_coordinates(
            input_index,
            value.shape,
        )
        if keepdims:
            output_coordinates = tuple(
                0 if dimension in axes_set else coordinate
                for dimension, coordinate in enumerate(input_coordinates)
            )
        else:
            output_coordinates = tuple(
                coordinate
                for dimension, coordinate in enumerate(input_coordinates)
                if dimension not in axes_set
            )
            if scalar_output_as_vector:
                output_coordinates = (0,)
        groups[
            coordinates_to_linear_index(output_coordinates, output_shape)
        ].append(input_index)

    return axes, output_shape, groups


def keepdims_shape(
    shape: tuple[int, ...],
    axis: Axis,
) -> tuple[int, ...]:
    """Return the reduction output shape with all reduced axes retained."""
    return reduction_shape(shape, normalize_axes(len(shape), axis), True)


__all__ = [
    "Axis",
    "immutable_axis",
    "keepdims_shape",
    "normalize_axes",
    "reduction_groups",
    "reduction_shape",
    "reduction_size",
]
