"""Enough of the matrix to see that the harness and the suites still work.

This is the profile to run after changing a benchmark, not the one to quote.
Three short rounds cannot separate a small regression from scheduling noise,
and the sizes it keeps are chosen to exercise both ends of a curve rather
than to describe its shape: the smallest, where fixed overhead dominates, and
the middle, where a backend has started to pay off.
"""

from __future__ import annotations

from . import Profile

PROFILE = Profile(
    name="quick",
    description="both ends of each curve, float64 only, three short rounds",
    rounds=3,
    target_seconds=0.01,
    scales=frozenset({"tiny", "medium"}),
    dtypes=frozenset({"float64"}),
    #: The expensive end of every curve is what makes a full run long, so it
    #: is declined here regardless of what a workload would otherwise carry.
    ceilings={"python": 10_000, "numpy": 100_000, "cuda": 100_000},
)

__all__ = ["PROFILE"]
