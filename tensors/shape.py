"""Immutable logical tensor shape metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import SupportsIndex, overload


class Shape(tuple[int, ...]):
    """Logical extents of a tensor.

    Shape() represents a scalar. Zero-sized dimensions are valid, while
    negative, boolean, and non-integer dimensions are rejected.
    """

    __slots__ = ()

    def __new__(cls, *dimensions: int) -> Shape:
        normalized = tuple(dimensions)
        for dimension in normalized:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(
                    f"Invalid shape {normalized}: dimensions must be integers"
                )
            if dimension < 0:
                raise ValueError(
                    f"Invalid shape {normalized}: "
                    "dimensions must be non-negative integers"
                )
        return tuple.__new__(cls, normalized)

    @classmethod
    def from_iterable(cls, dimensions: Iterable[int]) -> Shape:
        """Build a shape from any iterable of dimensions."""
        if type(dimensions) is cls:
            return dimensions
        try:
            normalized = tuple(dimensions)
        except TypeError as exc:
            raise TypeError(
                "shape must be an iterable of non-negative integers"
            ) from exc
        return cls(*normalized)

    def broadcast_with(self, other: Shape | Iterable[int]) -> Shape:
        """Return the shared shape produced by NumPy-style broadcasting."""
        normalized_other = (
            other if isinstance(other, Shape) else Shape.from_iterable(other)
        )
        # Equal shapes and rank-zero operands dominate expression code, and
        # both resolve without inspecting individual dimensions.
        if tuple.__eq__(self, normalized_other) is True:
            return self
        if not normalized_other:
            return self
        if not self:
            return normalized_other

        dimensions: list[int] = []
        for dimension, other_dimension in zip(
            reversed(self),
            reversed(normalized_other),
        ):
            if dimension == other_dimension:
                dimensions.append(dimension)
            elif dimension == 1:
                dimensions.append(other_dimension)
            elif other_dimension == 1:
                dimensions.append(dimension)
            else:
                raise ValueError(
                    f"Shapes {self} and {normalized_other} cannot be broadcast"
                )

        longer_shape = (
            self if self.rank > normalized_other.rank else normalized_other
        )
        matched_dimensions = min(self.rank, normalized_other.rank)
        unmatched_dimensions = longer_shape.rank - matched_dimensions
        dimensions.extend(
            reversed(tuple.__getitem__(longer_shape, slice(unmatched_dimensions)))
        )
        dimensions.reverse()
        # Every dimension came from an already-validated Shape.
        return tuple.__new__(Shape, dimensions)

    def stretched_axes_from(self, other: Shape | Iterable[int]) -> tuple[int, ...]:
        """Return the axes a broadcast from ``other`` stretched to reach this.

        :meth:`broadcast_with` agrees the shape two operands broadcast to;
        this answers the reverse question about one of them. Given the shape a
        broadcast produced, it says which of that shape's axes one operand did
        not have, or had as a singleton — the axes along which one of the
        operand's values fed several positions.

        That is the question a reverse pass asks: a value that fed several
        positions is owed the sum of them, so these are the axes to sum over
        and the only ones. Nothing here reads an element or mentions a
        gradient; it is the arithmetic of two shapes.

        Shapes align from the right, as they do when broadcasting forwards, so
        an axis ``other`` does not have at all is stretched like a singleton::

            Shape(2, 3).stretched_axes_from((1, 3))  # (0,)
            Shape(2, 3).stretched_axes_from((3,))    # (0,)
            Shape(2, 3).stretched_axes_from((2, 1))  # (1,)
            Shape(2, 3).stretched_axes_from((2, 3))  # ()

        Raises:
            ValueError: If ``other`` did not broadcast to this shape, either
                because it has more axes or because an axis disagrees without
                being a singleton. A caller that treats this as a capability
                question rather than an error catches it and declines.
        """
        operand = other if isinstance(other, Shape) else Shape.from_iterable(other)
        if operand.rank > self.rank:
            raise ValueError(f"Shape {operand} did not broadcast to {self}")
        padded = (1,) * (self.rank - operand.rank) + tuple(operand)
        axes = []
        for axis, (produced, original) in enumerate(zip(self, padded)):
            if produced == original:
                continue
            if original != 1:
                raise ValueError(f"Shape {operand} did not broadcast to {self}")
            axes.append(axis)
        return tuple(axes)

    @overload
    def __getitem__(self, key: SupportsIndex) -> int: ...

    @overload
    def __getitem__(self, key: slice) -> Shape: ...

    def __getitem__(self, key: SupportsIndex | slice) -> int | Shape:
        """Return one dimension or a Shape containing a dimension slice."""
        if isinstance(key, slice):
            return tuple.__new__(Shape, tuple.__getitem__(self, key))
        return tuple.__getitem__(self, key)

    @property
    def rank(self) -> int:
        """Number of logical dimensions."""
        return len(self)

    @property
    def size(self) -> int:
        """Number of logical elements described by this shape."""
        size = 1
        for dimension in self:
            size *= dimension
        return size


__all__ = ["Shape"]
