"""High-precision references for the power derivatives (section 12.7.4).

Section 12.7.4 asks for ordinary differentiable cases to be validated against
a high-precision reference, and it is explicit that the 2 and 4 ULP bounds of
section 12.6.2 are **not** imposed on a complete gradient expression: a
gradient composes a power, a logarithm and two multiplications, so error
necessarily accumulates. These functions therefore say what the derivative is;
they do not certify how close an implementation must come. The caller states
its own tolerance and says why.

The machinery is the enclosure of :mod:`_reference`, extended by one exact
rational factor:

* ``d/dx = y * x**(y-1)`` is ``scale * x**power`` with ``scale = y`` and
  ``power = y - 1`` computed exactly as rationals;
* ``d/dy = x**y * ln x`` is a power times a logarithm, and the logarithm's own
  interval is carried through rather than collapsed to a number.

Both refine the working precision until the product's two endpoints round to
the same representable value, and both refuse to answer otherwise.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from fractions import Fraction

from ._reference import (
    PRECISIONS,
    UnresolvedReference,
    _decimal_interval,
    _to_decimal,
    round_fraction,
)


def _as_fraction(value: float) -> Fraction:
    return Fraction(*value.as_integer_ratio())


def _power_interval(
    base: float, power: Fraction, digits: int
) -> tuple[Fraction, Fraction]:
    """An interval containing ``|base| ** power``, at one working precision."""
    with localcontext() as context:
        context.prec = digits
        context.Emax = 10**9
        context.Emin = -(10**9)
        logarithm = Decimal(abs(base)).ln()
    low, high = _decimal_interval(logarithm, digits)
    ends = (power * low, power * high)
    product_low, product_high = min(ends), max(ends)

    with localcontext() as context:
        context.prec = digits
        context.Emax = 10**9
        context.Emin = -(10**9)
        lower = _to_decimal(product_low, digits, "ROUND_FLOOR").exp()
        upper = _to_decimal(product_high, digits, "ROUND_CEILING").exp()
    return _decimal_interval(lower, digits)[0], _decimal_interval(upper, digits)[1]


def _logarithm_interval(base: float, digits: int) -> tuple[Fraction, Fraction]:
    """An interval containing ``ln(base)`` for a positive base."""
    with localcontext() as context:
        context.prec = digits
        context.Emax = 10**9
        context.Emin = -(10**9)
        logarithm = Decimal(base).ln()
    return _decimal_interval(logarithm, digits)


def _resolve(build, dtype: str, description: str) -> float:
    """Refine the precision until both endpoints round to the same value."""
    for digits in PRECISIONS:
        low, high = build(digits)
        if low > high:
            low, high = high, low
        rounded_low = round_fraction(low, dtype)
        rounded_high = round_fraction(high, dtype)
        if rounded_low == rounded_high and math.copysign(
            1.0, rounded_low
        ) == math.copysign(1.0, rounded_high):
            return rounded_low
    raise UnresolvedReference(
        f"{description} in {dtype} did not resolve at {PRECISIONS[-1]} digits"
    )


def base_derivative(base: float, exponent: float, dtype: str) -> float:
    """``y * x**(y-1)`` rounded once into ``dtype``.

    Only the ordinary rows of section 12.7.2 are supported; the classified
    rows are literals and need no reference.
    """
    if base == 0.0 or not math.isfinite(base) or not math.isfinite(exponent):
        raise ValueError("only the ordinary rows have a computed derivative")
    scale = _as_fraction(exponent)
    power = _as_fraction(exponent) - 1
    if base < 0.0 and not float(exponent).is_integer():
        raise ValueError("no real derivative for a negative base here")

    # The sign: y * x**(y-1), where x**(y-1) is negative exactly when x is
    # negative and y - 1 is an odd integer.
    sign = -1 if scale < 0 else 1
    if base < 0.0 and int(exponent - 1) % 2:
        sign = -sign
    magnitude = abs(scale)

    if float(exponent).is_integer() and abs(exponent) <= 512:
        exact = magnitude * abs(_as_fraction(base)) ** int(power)
        return sign * round_fraction(exact, dtype)

    def build(digits):
        low, high = _power_interval(base, power, digits)
        return sign * magnitude * low, sign * magnitude * high

    return _resolve(build, dtype, f"d/dx of {base!r} ** {exponent!r}")


def exponent_derivative(base: float, exponent: float, dtype: str) -> float:
    """``x**y * ln x`` rounded once into ``dtype``, for a positive base."""
    if base <= 0.0 or not math.isfinite(base) or not math.isfinite(exponent):
        raise ValueError("only a positive base has a computed derivative")
    power = _as_fraction(exponent)

    def build(digits):
        power_low, power_high = _power_interval(base, power, digits)
        log_low, log_high = _logarithm_interval(base, digits)
        ends = [a * b for a in (power_low, power_high) for b in (log_low, log_high)]
        return min(ends), max(ends)

    return _resolve(build, dtype, f"d/dy of {base!r} ** {exponent!r}")


__all__ = ["base_derivative", "exponent_derivative"]
