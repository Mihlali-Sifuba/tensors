"""The whole matrix, at a sample length that is worth trusting.

This profile admits every size category and every dtype, so what it runs is
exactly what the suites declare and nothing is filtered out. It is the
reference: a number quoted without a profile is a standard number, and the
other two profiles are described by what they leave out of it.
"""

from __future__ import annotations

from . import Profile

PROFILE = Profile(
    name="standard",
    description="every case a suite declares, at five rounds per case",
    rounds=5,
    target_seconds=0.05,
)

__all__ = ["PROFILE"]
