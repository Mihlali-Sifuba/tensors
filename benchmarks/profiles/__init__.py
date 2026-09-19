"""How much of the benchmark matrix to run, named once.

A workload says *what* to measure and at which sizes that question is
interesting. A profile says how much of that to actually run today: how many
rounds, how long each sample should be, which backends and suites, which
dtypes, and which size categories.

The two are kept apart because they change for different reasons. A size
belongs in a workload's curve because the workload's cost changes there; it
is skipped today because there is no time to run it. Folding the second
reason into the first is what produced a dozen hand-tuned tuples across the
suites, each quietly deciding both.

A profile therefore *filters* a curve rather than supplying one. A suite
keeps its own sizes and dtypes, and ``standard`` admits all of them, so the
full matrix is what a suite declares and nothing else:

    for size in selected_sizes((1, 100, 10_000, 1_000_000)):
        ...

The active profile is process-wide for the duration of a run, in the same way
the selected backend is, because a factory is called deep inside the runner
and threading a profile through every signature would buy nothing.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: Names of the profiles that ship with the package.
PROFILE_NAMES: tuple[str, ...] = ("quick", "standard", "comprehensive")

#: Every element count falls in exactly one category, by magnitude. These are
#: the sizes a curve is built from, so a profile can decline the expensive end
#: of every curve at once instead of each suite deciding separately.
SCALE_BOUNDS: tuple[tuple[str, int | None], ...] = (
    ("tiny", 10),
    ("small", 1_000),
    ("medium", 100_000),
    ("large", 1_000_000),
    ("huge", None),
)

SCALE_NAMES: tuple[str, ...] = tuple(name for name, _ in SCALE_BOUNDS)


def scale_of(size: int) -> str:
    """Return the category ``size`` belongs to."""
    for name, upper in SCALE_BOUNDS:
        if upper is None or size <= upper:
            return name
    return SCALE_NAMES[-1]


@dataclass(frozen=True)
class Profile:
    """One named selection over the benchmark matrix.

    ``dtypes``, ``scales``, ``suites``, and ``backends`` are filters: an empty
    selection means no restriction, so a profile never introduces a case a
    suite did not declare.
    """

    name: str
    description: str
    #: Measured samples per case, one per interleaved round.
    rounds: int
    #: Calibration target for one sample.
    target_seconds: float
    #: Size categories to admit. Empty admits every category.
    scales: frozenset[str] = field(default_factory=frozenset)
    #: dtype names to admit. Empty admits every dtype.
    dtypes: frozenset[str] = field(default_factory=frozenset)
    #: Suites to run. Empty runs every registered suite.
    suites: tuple[str, ...] = ()
    #: Backends to measure. Empty measures every installed backend.
    backends: tuple[str, ...] = ()
    #: Whether to run the separate, untimed allocation pass.
    collect_memory: bool = False
    #: Per-backend element ceilings that override a workload's own. Empty
    #: leaves each workload's ceiling in force.
    ceilings: dict[str, int] = field(default_factory=dict)

    def admits_scale(self, size: int) -> bool:
        """Return whether an element count is in scope for this profile."""
        return not self.scales or scale_of(size) in self.scales

    def admits_dtype(self, dtype_name: str) -> bool:
        """Return whether a dtype is in scope for this profile."""
        return not self.dtypes or dtype_name in self.dtypes

    def ceiling_for(self, backend: str, declared: int) -> int:
        """Return the tighter of a workload's ceiling and the profile's."""
        limit = self.ceilings.get(backend)
        return declared if limit is None else min(declared, limit)


def load(name: str) -> Profile:
    """Return the profile a module defines, by name."""
    if name not in PROFILE_NAMES:
        raise KeyError(f"unknown profile {name!r}; expected one of {PROFILE_NAMES}")
    module = importlib.import_module(f"{__package__}.{name}")
    profile: Profile = module.PROFILE
    return profile


_active: Profile | None = None


def active() -> Profile:
    """Return the profile in force, defaulting to ``standard``.

    ``standard`` admits everything, so code that never sets a profile sees
    exactly the matrix its suites declare.
    """
    global _active
    if _active is None:
        _active = load("standard")
    return _active


@contextmanager
def use(profile: Profile | str) -> Iterator[Profile]:
    """Make ``profile`` the active one for the duration of the block."""
    global _active
    resolved = load(profile) if isinstance(profile, str) else profile
    previous = _active
    _active = resolved
    try:
        yield resolved
    finally:
        _active = previous


def selected_sizes(curve: Sequence[int]) -> tuple[int, ...]:
    """Return the part of a workload's size curve this profile admits."""
    profile = active()
    return tuple(size for size in curve if profile.admits_scale(size))


def selected_dtypes(dtypes: Sequence[str]) -> tuple[str, ...]:
    """Return the part of a workload's dtype selection this profile admits."""
    profile = active()
    return tuple(name for name in dtypes if profile.admits_dtype(name))


def settings() -> dict[str, Any]:
    """Return the active profile as something a stored report can carry."""
    profile = active()
    return {
        "name": profile.name,
        "description": profile.description,
        "rounds": profile.rounds,
        "target_seconds": profile.target_seconds,
        "scales": sorted(profile.scales) or list(SCALE_NAMES),
        "dtypes": sorted(profile.dtypes),
        "suites": list(profile.suites),
        "backends": list(profile.backends),
        "collect_memory": profile.collect_memory,
        "ceilings": dict(profile.ceilings),
    }


__all__ = [
    "PROFILE_NAMES",
    "SCALE_BOUNDS",
    "SCALE_NAMES",
    "Profile",
    "active",
    "load",
    "scale_of",
    "selected_dtypes",
    "selected_sizes",
    "settings",
    "use",
]
