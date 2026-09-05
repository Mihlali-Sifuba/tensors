"""Ordered weak-reference storage used by graph inspection registries."""

from __future__ import annotations

from typing import Generic, TypeVar
from weakref import ReferenceType, ref


T = TypeVar("T")


class WeakRegistry(Generic[T]):
    """Keep insertion order without extending the lifetime of registered objects."""

    __slots__ = ("_references", "_additions", "__weakref__")

    _MINIMUM_PRUNE_INTERVAL = 256

    def __init__(self) -> None:
        self._references: dict[int, ReferenceType[T]] = {}
        self._additions = 0

    def add(self, value: T) -> None:
        """Register ``value`` without taking ownership of it."""
        identity = id(value)
        existing = self._references.get(identity)
        if existing is not None and existing() is value:
            return
        if existing is not None:
            del self._references[identity]
        self._references[identity] = ref(value)
        self._additions += 1
        # Pruning walks every registration, so the interval scales with the
        # live population. A fixed interval would make recording a large
        # graph quadratic in the number of registered nodes.
        if self._additions >= max(
            self._MINIMUM_PRUNE_INTERVAL,
            len(self._references),
        ):
            self._prune()

    def __contains__(self, value: object) -> bool:
        """Return whether the identical live object is registered."""
        reference = self._references.get(id(value))
        return reference is not None and reference() is value

    def values(self) -> list[T]:
        """Return live values and discard dead weak references."""
        live = []
        dead = []
        for identity, reference in self._references.items():
            value = reference()
            if value is not None:
                live.append(value)
            else:
                dead.append(identity)
        for identity in dead:
            self._references.pop(identity, None)
        return live

    def _prune(self) -> None:
        """Periodically discard dead entries without per-value callbacks."""
        dead = [
            identity
            for identity, reference in self._references.items()
            if reference() is None
        ]
        for identity in dead:
            self._references.pop(identity, None)
        self._additions = 0

    def clear(self) -> None:
        """Forget every registration without modifying the registered objects."""
        self._references.clear()
        self._additions = 0
