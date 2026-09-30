"""An independent reference for the package's floating summation contract.

`docs/summation-semantics.md` specifies the reduction as a tree, not as the
exact sum: a group in logical order, adjacent pairs added with one rounding
to the declared dtype, an odd final value carried unchanged, repeated until one
value remains, followed by the non-finite classification and the canonical
zero. This module states that tree again from the specification so that tests
can compare every backend against it rather than against each other. It shares
no code with the implementation.

Binary32 rounding is done with ``struct``: a single sum or product of two
binary32 values is exact in binary64, so packing it to ``"f"`` performs exactly
the one correctly rounded binary32 operation the contract requires.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from fractions import Fraction

FLOAT32_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]


def round32(value: float) -> float:
    """Round one binary64 value to binary32, overflowing to infinity."""
    if math.isnan(value) or math.isinf(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def rounder(dtype_name: str):
    """Return the one-operation rounding for ``float32`` or ``float64``."""
    if dtype_name == "float32":
        return round32
    if dtype_name == "float64":
        return lambda value: value
    raise ValueError(f"not a floating dtype: {dtype_name}")


def pairwise_sum(values: Iterable[float], dtype_name: str = "float64") -> float:
    """The specified pairwise reduction of one group."""
    rnd = rounder(dtype_name)
    level = [rnd(float(value)) for value in values]
    if not level:
        return 0.0
    if any(math.isnan(value) for value in level):
        return math.nan
    positive = any(value == math.inf for value in level)
    negative = any(value == -math.inf for value in level)
    while len(level) > 1:
        paired = [rnd(level[i] + level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            paired.append(level[-1])
        level = paired
    if positive and negative:
        return math.nan
    if positive:
        return math.inf
    if negative:
        return -math.inf
    return 0.0 if level[0] == 0.0 else level[0]


def pairwise_dot(
    left: Sequence[float], right: Sequence[float], dtype_name: str = "float64"
) -> float:
    """Products rounded to the dtype, then the specified pairwise reduction."""
    rnd = rounder(dtype_name)
    return pairwise_sum(
        [rnd(float(a) * float(b)) for a, b in zip(left, right, strict=True)],
        dtype_name,
    )


def left_to_right_sum(values: Iterable[float]) -> float:
    """Sequential binary64 summation, for tests that distinguish the orders."""
    total = 0.0
    for value in values:
        total += float(value)
    return total


def error_bound(values: Sequence[float], dtype_name: str) -> Fraction:
    """The documented bound on |pairwise - exact| without overflow.

    It is evaluated exactly: in binary64, ``1 + 2**-53`` is already 1, so a
    floating evaluation of ``(1 + u) ** k - 1`` would report a zero bound.
    """
    n = len(values)
    if n <= 1:
        return Fraction(0)
    unit = Fraction(1, 2**24) if dtype_name == "float32" else Fraction(1, 2**53)
    tiny = Fraction(1, 2**149) if dtype_name == "float32" else Fraction(1, 2**1074)
    depth = (n - 1).bit_length()
    growth = (1 + unit) ** depth
    magnitude = sum((abs(Fraction(value)) for value in values), Fraction(0))
    return (growth - 1) * magnitude + (n - 1) * (tiny / 2) * growth
