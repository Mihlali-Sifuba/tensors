"""Everything, measured long enough to be quoted, including allocation.

This profile differs from ``standard`` in how hard it looks rather than in
what it looks at: more rounds and longer samples, so the median is steadier
and the noise verdict means more, plus the separate allocation pass for the
cases that ask for one. It is the profile a published figure comes from.
"""

from __future__ import annotations

from . import Profile

PROFILE = Profile(
    name="comprehensive",
    description="every case, nine rounds, longer samples, with memory",
    rounds=9,
    target_seconds=0.1,
    collect_memory=True,
)

__all__ = ["PROFILE"]
