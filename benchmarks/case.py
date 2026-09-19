"""What a benchmark measures, and how several of them are grouped.

A :class:`Case` names one computation, the layer of the execution stack it is
taken at, and the backends it means anything on. It carries no timing: a case
is a description, and measuring it is the scheduler's job.

A :class:`Group` is the set of cases whose inputs are built and released
together. Grouping bounds live memory, because only one group's inputs exist
at a time across every backend it covers, and it is also the interleaving
unit, so the cases measured close together are the ones meant to be compared.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

Backend: TypeAlias = Literal["python", "numpy", "cuda"]

#: Where in the execution stack a case measures. Comparing adjacent layers
#: over the same computation is what attributes overhead to a layer.
Layer: TypeAlias = Literal[
    "provider",
    "kernel",
    "dispatch",
    "public",
    "variable",
    "graph-trace",
    "graph-compile",
    "graph-replay",
    "autograd",
    "optimizer",
    "training",
    "storage",
    "fusion",
    "startup",
    "sync",
    "memory",
]

Classification: TypeAlias = Literal[
    "measured",
    "unsupported",
    "skipped",
    "error",
]


class Unsupported(Exception):
    """Raised by a case factory when a backend cannot express the case."""


@dataclass(frozen=True)
class Case:
    """One independently calibrated measurement.

    ``run`` must perform exactly the work being attributed to ``layer``.
    Anything shared, reusable, or merely preparatory belongs in the factory
    that builds the case, or in ``setup``.
    """

    name: str
    run: Callable[[], Any]
    layer: Layer = "public"
    family: str = "misc"
    #: Backends this case is meaningful on. ``None`` means every backend.
    backends: frozenset[str] | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | str | None = None
    elements: int | None = None
    work_items: int | None = None
    validate: Callable[[], None] | None = None
    setup: Callable[[], None] | None = None
    reset: Callable[[], None] | None = None
    teardown: Callable[[], None] | None = None
    #: A case that builds cyclic objects each call keeps collection in scope.
    gc_enabled: bool = False
    #: A single state transition per sample; calibration must not batch it.
    single_shot: bool = False
    #: Include this case in the separate memory pass.
    memory: bool = False
    description: str = ""
    #: Free-form comparison keys, e.g. ``{"op": "add", "against": "..."}``.
    tags: dict[str, str] = field(default_factory=dict)

    def supports(self, backend: str) -> bool:
        """Return whether this case is meaningful for ``backend``."""
        return self.backends is None or backend in self.backends


#: A factory receives the active backend and returns the cases it can build
#: there. Raising :class:`Unsupported` classifies the whole group explicitly.
CaseFactory: TypeAlias = Callable[[str], Sequence[Case]]


@dataclass(frozen=True)
class Group:
    """Cases whose inputs are built and released together.

    Grouping bounds live memory: only one group's inputs exist at a time,
    across every backend it covers. It is also the interleaving unit, so the
    cases measured close together are the ones meant to be compared.
    """

    name: str
    factory: CaseFactory
    suite: str


__all__ = [
    "Backend",
    "Case",
    "CaseFactory",
    "Classification",
    "Group",
    "Layer",
    "Unsupported",
]
