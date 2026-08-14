"""Ordered weak-reference storage used by graph inspection registries."""

from __future__ import annotations

from typing import Generic, TypeVar
from weakref import ReferenceType, ref


T = TypeVar("T")


class WeakRegistry(Generic[T]):
    """Keep insertion order without extending the lifetime of registered objects."""

    __slots__ = ("_references", "__weakref__")

    def __init__(self) -> None:
        self._references: list[ReferenceType[T]] = []

    def add(self, value: T) -> None:
        """Register ``value`` without taking ownership of it."""
        owner_reference = ref(self)

        def discard(reference: ReferenceType[T]) -> None:
            owner = owner_reference()
            if owner is None:
                return
            try:
                owner._references.remove(reference)
            except ValueError:
                pass

        self._references.append(ref(value, discard))

    def values(self) -> list[T]:
        """Return live values and discard dead weak references."""
        live = []
        references = []
        for reference in self._references:
            value = reference()
            if value is not None:
                live.append(value)
                references.append(reference)
        if len(references) != len(self._references):
            self._references = references
        return live

    def clear(self) -> None:
        """Forget every registration without modifying the registered objects."""
        self._references.clear()
