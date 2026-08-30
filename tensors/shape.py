"""Immutable logical tensor shape metadata."""

from __future__ import annotations

from collections.abc import Iterable


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
        try:
            normalized = tuple(dimensions)
        except TypeError as exc:
            raise TypeError(
                "shape must be an iterable of non-negative integers"
            ) from exc
        return cls(*normalized)

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
