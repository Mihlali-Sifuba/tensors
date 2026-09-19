"""Where the suites are, and what kind of thing each one measures.

A suite name is what a reader selects on the command line and what a stored
record is labelled with, so the names here are the vocabulary of the whole
system. They are grouped by the three questions the benchmarks answer:

``workloads``
    What does this operation cost, and where in the stack does that cost
    enter? Organized by what the operation means, never by backend.
``execution``
    What does the machinery around an operation cost on its own: recording a
    graph, compiling it, replaying it, differentiating it, moving storage.
``scenarios``
    What does a whole workflow cost, with every layer included at once.

Suites are imported lazily, so selecting one narrow suite does not pay for
building every other suite's groups.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import replace

from .case import Group

#: Semantic operations, each owning its full measurement ladder.
WORKLOAD_SUITES: dict[str, str] = {
    "arithmetic": "workloads.arithmetic",
    "elementary": "workloads.elementary",
    "trigonometric": "workloads.trigonometric",
    "hyperbolic": "workloads.hyperbolic",
    "activations": "workloads.activations",
    "comparison": "workloads.comparison",
    "reductions": "workloads.reductions",
    "manipulation": "workloads.manipulation",
    "layout": "workloads.layout",
    "linalg": "workloads.linalg",
    "broadcasting": "workloads.broadcasting",
    "losses": "workloads.losses",
    "convolution": "workloads.convolution",
    "creation": "workloads.creation",
    "random": "workloads.random",
    "initializers": "workloads.initializers",
}

#: Framework behaviour that can be measured without a semantic operand.
EXECUTION_SUITES: dict[str, str] = {
    "dispatch": "execution.dispatch",
    "graph": "execution.graph",
    "backward": "execution.backward",
    "fusion": "execution.fusion",
    "storage": "execution.storage",
    "synchronization": "execution.synchronization",
    "startup": "execution.startup",
    "threading": "execution.threading",
    "allocation": "execution.allocation",
}

#: Whole workflows, which include every layer on purpose.
SCENARIO_SUITES: dict[str, str] = {
    "training": "scenarios.training_step",
    "optimizer": "scenarios.optimizer",
}

#: Suite name to the module providing its ``groups()`` factory.
SUITE_MODULES: dict[str, str] = {
    **WORKLOAD_SUITES,
    **EXECUTION_SUITES,
    **SCENARIO_SUITES,
}

#: The suites that together answer the questions the map was built for.
DEFAULT_SUITES: tuple[str, ...] = tuple(SUITE_MODULES)

#: A short selection for verifying the harness itself.
SMOKE_SUITES: tuple[str, ...] = ("arithmetic", "reductions", "dispatch")


def kind_of(name: str) -> str:
    """Return whether a suite is a workload, execution machinery, or a scenario."""
    if name in WORKLOAD_SUITES:
        return "workload"
    if name in EXECUTION_SUITES:
        return "execution"
    if name in SCENARIO_SUITES:
        return "scenario"
    raise KeyError(f"unknown suite {name!r}")


def suite_groups(name: str) -> Sequence[Group]:
    """Return the groups one suite defines, labelled with its registered name.

    The suite name is stamped here rather than repeated inside each module,
    so the name a reader selects on the command line and the name a stored
    record carries cannot drift apart.
    """
    if name not in SUITE_MODULES:
        raise KeyError(f"unknown suite {name!r}")
    module = importlib.import_module(f"{__package__}.{SUITE_MODULES[name]}")
    factory: Callable[[], Sequence[Group]] = module.groups
    return [replace(group, suite=name) for group in factory()]


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


__all__ = [
    "DEFAULT_SUITES",
    "EXECUTION_SUITES",
    "SCENARIO_SUITES",
    "SMOKE_SUITES",
    "SUITE_MODULES",
    "WORKLOAD_SUITES",
    "collect",
    "kind_of",
    "suite_groups",
]
