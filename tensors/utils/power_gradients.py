"""The region table for the derivatives of ``x ** y`` (section 12.7.2).

`docs/arithmetic-semantics.md` section 12.7 states each partial derivative as a
**classified** value: it exists, it does not exist and NaN records that, or no
finite derivative exists and an approved convention represents a one-sided
infinite slope. The formulas

.. math:: \\partial f/\\partial x = y x^{y-1}, \\qquad
          \\partial f/\\partial y = x^{y} \\ln x

are not valid as unconditionally evaluated expressions, and evaluating them
and hoping IEEE arrives at the right answer does not work: at ``x = 0`` with
``0 < y < 1`` the first gives ``0 * inf``, and at ``x = 0`` with ``y > 0`` the
second gives ``0 * (-inf)``. Both are NaN in IEEE, and neither is the
specified result. The table is therefore applied, not inferred.

This module holds the scalar form, shared by the Python backend kernels and by
the second-order rules in :mod:`tensors.operations.arithmetic.power`. The
array backends state the same table over masks, in their own kernels.

Nothing here raises on a numerical condition (rule G2), and nothing here reads
a tensor (rule G3): the callers pass scalars they already hold.
"""

from __future__ import annotations

import math

_INFINITY = math.inf
_NAN = math.nan

#: The three classes of section 12.7.2. ``CONVENTION`` is not a derivative: it
#: is an approved representation of a one-sided infinite limiting slope, and
#: the approval covers exactly one region.
EXISTS = "exists"
NAN = "nan"
CONVENTION = "convention"


def is_integral(value: float) -> bool:
    """Whether an exponent is an integer, so a negative base stays real."""
    return math.isfinite(value) and float(value).is_integer()


def base_derivative(base: float, exponent: float) -> tuple[float, str]:
    """Return ``(d/dx of x**y, its class)`` at one point (section 12.7.2).

    The ordinary formula is returned as ``None`` in the value slot when the
    caller should evaluate it stably itself; every other row is a literal.
    """
    if base != base or exponent != exponent:
        return (_NAN, NAN)  # IEEE propagation
    if base == 0.0:
        # -0.0 compares equal to 0.0, so it takes these rows too. The table
        # states +inf without qualification and no sign variant is approved.
        if exponent == 0.0:
            return (0.0, EXISTS)  # f(x, 0) = 1 is constant in x
        if exponent < 0.0:
            return (_NAN, NAN)
        if exponent == 1.0:
            return (1.0, EXISTS)
        if exponent > 1.0:
            return (0.0, EXISTS)
        return (_INFINITY, CONVENTION)  # 0 < y < 1
    if base < 0.0 and not is_integral(exponent):
        return (_NAN, NAN)
    return (None, EXISTS)


def exponent_derivative(base: float, exponent: float) -> tuple[float, str]:
    """Return ``(d/dy of x**y, its class)`` at one point (section 12.7.2)."""
    if base != base or exponent != exponent:
        return (_NAN, NAN)  # IEEE propagation
    if base == 0.0:
        if exponent > 0.0:
            # f(0, y) = 0 for every y > 0, so f is constant in y and the
            # derivative is exactly zero. The formula's 0 * (-inf) is a
            # degenerate encoding of a derivative that genuinely exists.
            return (0.0, EXISTS)
        # y = 0: f(0, y) is 1 at 0 and 0 above it, so it is discontinuous.
        # y < 0: the forward result is an infinity, which establishes nothing
        # about a derivative at the evaluation point.
        return (_NAN, NAN)
    if base < 0.0:
        return (_NAN, NAN)  # ln x is undefined for x < 0, for every y
    return (None, EXISTS)


__all__ = [
    "CONVENTION",
    "EXISTS",
    "NAN",
    "base_derivative",
    "exponent_derivative",
    "is_integral",
]
