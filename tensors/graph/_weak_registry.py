"""Ordered weak-reference storage used by graph inspection registries."""

from __future__ import annotations

from typing import Generic, TypeVar
from weakref import ReferenceType, ref


T = TypeVar("T")


class WeakRegistry(Generic[T]):
    """Keep insertion order without extending the lifetime of registered objects."""

    __slots__ = ("_references", "__weakref__")

    def __init__(self) -> None:
        self._references: dict[int, ReferenceType[T]] = {}

    def add(self, value: T) -> None:
        """Register ``value`` without taking ownership of it."""
        identity = id(value)
        existing = self._references.get(identity)
        if existing is not None and existing() is value:
            return
        if existing is not None:
            del self._references[identity]
        owner_reference = ref(self)

        def discard(reference: ReferenceType[T]) -> None:
            owner = owner_reference()
            if owner is None:
                return
            if owner._references.get(identity) is reference:
                del owner._references[identity]

        reference = ref(value, discard)
        self._references[identity] = reference

    def __contains__(self, value: object) -> bool:
        """Return whether the identical live object is registered."""
        reference = self._references.get(id(value))
        return reference is not None and reference() is value

    def values(self) -> list[T]:
        """Return live values and discard dead weak references."""
        live = []
        dead = []
        for identity, reference in tuple(self._references.items()):
            value = reference()
            if value is not None:
                live.append(value)
            else:
                dead.append(identity)
        for identity in dead:
            self._references.pop(identity, None)
        return live

    def clear(self) -> None:
        """Forget every registration without modifying the registered objects."""
        self._references.clear()
