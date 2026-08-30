"""Immutable physical tensor stride metadata."""

from __future__ import annotations

from collections.abc import Iterable

from .shape import Shape


class Strides(tuple[int, ...]):
    """Physical storage movement for one logical step along each axis.

    Strides may be positive, zero, or negative. Coordinate validity is a
    property of a shape and is intentionally not enforced here.
    """

    __slots__ = ()

    def __new__(cls, *values: int) -> Strides:
        normalized = tuple(values)
        for stride in normalized:
            if isinstance(stride, bool) or not isinstance(stride, int):
                raise TypeError("strides must contain only integers")
        return tuple.__new__(cls, normalized)

    @classmethod
    def from_iterable(cls, values: Iterable[int]) -> Strides:
        """Build strides from any iterable of integer values."""
        try:
            normalized = tuple(values)
        except TypeError as exc:
            raise TypeError("strides must be an iterable of integers") from exc
        return cls(*normalized)

    @classmethod
    def contiguous(cls, shape: Shape | Iterable[int]) -> Strides:
        """Return canonical row-major element strides for shape.

        Scalars have no strides. For zero-sized shapes, trailing products are
        retained deliberately; Shape(2, 0, 3) has Strides(0, 3, 1).
        """
        normalized = (
            shape if isinstance(shape, Shape) else Shape.from_iterable(shape)
        )
        stride = 1
        values: list[int] = []
        for dimension in reversed(normalized):
            values.append(stride)
            stride *= dimension
        return cls(*reversed(values))


__all__ = ["Strides"]
