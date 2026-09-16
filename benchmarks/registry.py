"""The suite registry.

Suites are imported lazily so that selecting one narrow suite does not pay
for building every other suite's groups.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

from .case import Group


#: Suite name to the module providing its ``groups()`` factory.
SUITE_MODULES: dict[str, str] = {
    "layers": "layers",
    "reductions": "reductions",
    "broadcasting": "broadcasting",
    "layout": "layout",
    "linalg": "linalg",
    "shape": "shape",
    "creation": "creation",
    "random": "randomness",
    "convolution": "convolution",
    "losses": "losses",
    "initializers": "initializers",
    "storage": "storage",
    "framework": "framework",
    "graph": "graph",
    "autograd": "autograd",
    "optimizer": "optimizer",
    "training": "training",
    "threading": "threading",
    "fusion": "fusion",
    "cuda": "cudasync",
    "startup": "startup",
    "memory": "memoryprofile",
}

#: The suites that together answer the questions the map was built for.
DEFAULT_SUITES: tuple[str, ...] = tuple(SUITE_MODULES)

#: A short selection for verifying the harness itself.
SMOKE_SUITES: tuple[str, ...] = ("layers", "reductions", "framework")


def suite_groups(name: str) -> Sequence[Group]:
    """Return the groups one suite defines."""
    if name not in SUITE_MODULES:
        raise KeyError(f"unknown suite {name!r}")
    module = importlib.import_module(
        f"{__package__}.suites.{SUITE_MODULES[name]}"
    )
    factory: Callable[[], Sequence[Group]] = module.groups
    return factory()


def collect(
    names: Sequence[str],
    *,
    match: str | None = None,
) -> list[Group]:
    """Return the groups for the selected suites, optionally filtered."""
    groups: list[Group] = []
    for name in names:
        for group in suite_groups(name):
            if match and match.casefold() not in group.name.casefold():
                continue
            groups.append(group)
    return groups


__all__ = ["DEFAULT_SUITES", "SMOKE_SUITES", "SUITE_MODULES", "collect",
           "suite_groups"]
