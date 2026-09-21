"""A validated independent reference for the correctly rounded square root.

`docs/sqrt-semantics.md` section 5. Accuracy is measured against this module,
never against another backend.

The module answers one question: **what is the correctly rounded square root
of a finite positive value in a declared format?** Nothing here is used by
ordinary tensor execution.

The procedure is exact integer arithmetic throughout, and needs no
high-precision decimal arithmetic at all, because a square root can be
bracketed exactly:

1. The operand is a binary float, hence exactly a rational ``p / q``.
2. ``sqrt(p / q)`` equals ``sqrt(p * q) / q``, so one integer square root of
   ``p * q`` scaled by ``4 ** k`` gives ``r`` with
   ``r / (q * 2**k) <= sqrt(p / q) < (r + 1) / (q * 2**k)``. The bracket is
   exact, and its width shrinks by a factor of two for every added bit of
   ``k``.
3. Each endpoint is rounded **directly from its rational value** into the
   target format by :func:`round_fraction`, so a ``float32`` reference rounds
   once rather than through ``float64``.

The result is accepted only when both endpoints round to the same
representable value; otherwise ``k`` is raised and the evaluation repeats. A
true root exactly on a rounding boundary would never resolve — but that
cannot happen here, because a root that lands exactly on a representable
value or a midpoint is rational, and the perfect-square case below returns it
directly.

:func:`round_fraction` is reused from the arithmetic reference rather than
reimplemented: it is the same validated rounding
`docs/arithmetic-semantics.md` section 12.6.5 relies on, and it already
handles subnormals, ties to even and overflow.
"""

from __future__ import annotations

import math
from fractions import Fraction

from tests.operations.arithmetic._reference import FORMATS, round_fraction

#: Bracket precisions tried in turn, in bits.
_BRACKET_BITS: tuple[int, ...] = (64, 128, 256, 512, 1024)


class UnresolvedSquareRoot(Exception):
    """The bracket did not resolve the rounding at the largest precision."""


def reference_sqrt(value: float, dtype: str) -> float:
    """The correctly rounded square root of ``value`` in ``dtype``.

    ``value`` must be finite and non-negative; the specification's NaN and
    infinity rules are classifications rather than roundings and are asserted
    directly by the tests.
    """
    if dtype not in FORMATS:
        raise ValueError(f"unsupported reference format: {dtype!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"reference_sqrt needs a finite non-negative value: {value!r}")
    if value == 0.0:
        # Section 1.3: the sign of zero is carried through unchanged.
        return value

    exact = Fraction(*float(value).as_integer_ratio())
    numerator, denominator = exact.numerator, exact.denominator
    for bits in _BRACKET_BITS:
        scaled = (numerator * denominator) << (2 * bits)
        root = math.isqrt(scaled)
        if root * root == scaled:
            # The root is rational and exactly bracketed by itself.
            return round_fraction(Fraction(root, denominator << bits), dtype)
        below = round_fraction(Fraction(root, denominator << bits), dtype)
        above = round_fraction(Fraction(root + 1, denominator << bits), dtype)
        if below == above:
            return below
    raise UnresolvedSquareRoot(
        f"sqrt({value!r}) did not resolve in {dtype} at {_BRACKET_BITS[-1]} bits"
    )


__all__ = ["UnresolvedSquareRoot", "reference_sqrt"]
