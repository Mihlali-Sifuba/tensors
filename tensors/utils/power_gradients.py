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

This module holds the scalar form, shared by the Python backend kernels for
the first derivatives and for the three second partials. The array backends
state the same table over masks, in their own kernels.

:func:`power_product` is here for the same reason. Every one of those
derivatives is a few small factors times a power, and it is the power that
leaves the representable range while the whole expression stays inside it, so
the grouping that protects it is one computation stated once rather than six
copies that could drift.

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


def _real_power(base: float, exponent: float) -> float:
    """Return ``base ** exponent`` as a real value, or an infinity.

    ``math.pow`` raises for a domain error and for an overflow; neither is a
    condition this module may report as an exception (rule G2), so both come
    back as the value IEEE would give. A domain error is NaN and an overflow
    is a signed infinity.
    """
    try:
        return math.pow(base, exponent)
    except OverflowError:
        return _INFINITY if base > 0.0 or is_integral(exponent) else _NAN
    except ValueError:
        # ``pow(0, negative)`` is a pole, which IEEE reports as an infinity;
        # a negative base with a non-integral exponent has no real value.
        if base == 0.0:
            return _INFINITY
        return _NAN


def product_quotient(
    numerators: list[float], denominators: list[float] | None = None
) -> float:
    """Evaluate a product quotient without avoidable range loss.

    Multiplying left to right can overflow or underflow on an intermediate
    where the result is representable. Every finite float is a rational, so
    the factors are combined exactly as one ratio of integers and divided
    once, which rounds once and only at the end.
    """
    denominators = [] if denominators is None else denominators
    if any((math.isnan(value) for value in numerators + denominators)):
        return _NAN
    if any((value == 0.0 for value in denominators)):
        raise ZeroDivisionError("Division by zero")
    if any((value == 0.0 for value in numerators)):
        return 0.0
    if all((math.isfinite(value) for value in numerators + denominators)):
        numerator = 1
        denominator = 1
        for value in numerators:
            value_numerator, value_denominator = value.as_integer_ratio()
            numerator *= value_numerator
            denominator *= value_denominator
        for value in denominators:
            value_numerator, value_denominator = value.as_integer_ratio()
            numerator *= value_denominator
            denominator *= value_numerator
        try:
            return numerator / denominator
        except OverflowError:
            return _INFINITY if numerator * denominator > 0 else -_INFINITY
    result = 1.0
    for value in numerators:
        result *= value
    for value in denominators:
        result /= value
    return result


def power_product(factors: list[float], base: float, exponent: float) -> float:
    """Return ``product(factors) * base ** exponent`` without range loss.

    Every second partial derivative of ``x ** y`` has this shape — a few
    small factors times a power — and the power is the part that leaves the
    representable range while the product does not. Forming it first would
    round to an infinity or to nothing and lose a result that exists, so the
    power is used directly only where it is finite and non-zero, and the
    logarithms carry the magnitude otherwise.

    An exactly zero factor short-circuits to zero. That is a range guard and
    also the reason ``y = 0`` and ``y = 1`` give an exactly zero second
    derivative rather than the ``0 * inf`` the unguarded formula produces.

    Nothing here raises on a numerical condition (rule G2).
    """
    if any((value == 0.0 for value in factors)):
        return 0.0
    power = _real_power(base, exponent)
    if power != 0.0 and math.isfinite(power):
        return product_quotient(factors + [power])
    if (
        base != 0.0
        and all((math.isfinite(value) for value in factors))
        and math.isfinite(base)
        and math.isfinite(exponent)
    ):
        sign = -1.0 if sum((value < 0.0 for value in factors)) % 2 else 1.0
        if base < 0.0:
            if not is_integral(exponent):
                # Section 12.3.3 gives NaN rather than an error here.
                return _NAN
            if int(exponent) % 2:
                sign = -sign
        logarithm = math.fsum(
            [math.log(abs(value)) for value in factors]
            + [exponent * math.log(abs(base))]
        )
        try:
            magnitude = math.exp(logarithm)
        except OverflowError:
            magnitude = _INFINITY
        return math.copysign(magnitude, sign)
    return product_quotient(factors + [power])


__all__ = [
    "CONVENTION",
    "EXISTS",
    "NAN",
    "base_derivative",
    "exponent_derivative",
    "is_integral",
    "power_product",
    "product_quotient",
]
