"""A validated high-precision reference for floating-point exponentiation.

`docs/arithmetic-semantics.md` section 12.6.5. Accuracy is measured against
this module, never against another backend.

The module answers one question: **what is the correctly rounded value of
``x ** y`` in a declared format?** It answers it only when it can prove the
answer, and reports the case unresolved otherwise. Nothing here is used by
ordinary tensor execution; arbitrary-precision arithmetic belongs to the test
infrastructure alone.

Two procedures are used.

*Exact rational.* For an integral exponent, ``x ** n`` is a rational number
and is computed as one. ``x`` is a binary float, hence exactly a rational, and
integer exponentiation of a rational is exact. No enclosure is needed because
no error is committed. The procedure is used while the exact numerator and
denominator stay within a size limit.

*Enclosure.* Otherwise ``exp(y * ln x)`` is evaluated as an **interval**. Each
step carries endpoints that provably bracket the true value, so the final
interval does too:

1. ``ln x`` is computed by :meth:`decimal.Decimal.ln`, which is documented as
   correctly rounded, and is then widened by one unit in its last place. The
   widened interval contains ``ln x``.
2. ``y * ln x`` multiplies that interval by the exact rational ``y``. Rational
   multiplication commits no error, so the product interval is exact given its
   input interval. This is where the *absolute* error of the logarithm is
   amplified by ``|y|`` — carrying an absolute rather than a relative bound on
   ``ln x`` is what makes the amplification explicit.
3. ``exp`` is increasing, so the image of an interval is the interval of the
   images. Each endpoint is converted to a decimal with *directed* rounding —
   down for the lower end, up for the upper — so the conversion cannot narrow
   the interval, and :meth:`decimal.Decimal.exp` is then widened by one unit
   in its last place as in step 1. This is where the amplification of
   ``|y ln x|`` lands: an interval of width ``w`` around ``y ln x`` becomes one
   of relative width about ``w``, so a large ``|y ln x|`` needs a
   correspondingly narrow logarithm, which the precision loop supplies.

Both endpoints are then rounded **directly from their rational values** into
the target format, so a ``float32`` reference rounds once rather than through
``float64``. The result is accepted only when both endpoints round to the same
representable value; otherwise the working precision is raised and the whole
evaluation repeats. If the largest supported precision does not resolve the
rounding, the case is reported unresolved and never guessed.

That final agreement test is what makes the procedure sound. No fixed
precision is provably sufficient — a true value arbitrarily close to a
rounding boundary needs arbitrarily more precision to resolve, and the worst
cases of ``pow`` are catalogued for none of the libraries involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

#: Section 3.2 format parameters. ``precision`` counts the significand bits,
#: the leading one included; ``min_exponent`` is the exponent of the smallest
#: normal value, which fixes the subnormal quantum.
FORMATS: dict[str, dict[str, int]] = {
    "float32": {"precision": 24, "min_exponent": -126, "max_exponent": 127},
    "float64": {"precision": 53, "min_exponent": -1022, "max_exponent": 1023},
}

#: Working precisions tried in turn, in significant decimal digits.
PRECISIONS: tuple[int, ...] = (40, 80, 160, 320, 640, 1280, 2560)

#: The exact rational path is used while ``x ** n`` stays under this many bits
#: in numerator and denominator together. Beyond it the enclosure is cheaper.
EXACT_BIT_LIMIT = 1 << 18

#: ``ln`` of the largest finite binary64 value is about 709.78, and ``ln`` of
#: half the smallest binary64 subnormal is about -744.4. These thresholds sit
#: well outside both, so a product beyond them settles the result without
#: evaluating an exponential that would need an enormous decimal exponent.
CERTAIN_OVERFLOW = Fraction(720)
CERTAIN_UNDERFLOW = Fraction(-800)


class OutsideReferenceDomain(Exception):
    """The operand pair is not an ordinary case the ULP metric applies to."""


class UnresolvedReference(Exception):
    """The rounding could not be resolved at the largest supported precision.

    Raised rather than returning a guess. Section 12.6.5 step 5: never return
    an unvalidated reference.
    """


@dataclass(frozen=True)
class Reference:
    """A correctly rounded result and the evidence that it is one.

    ``low`` and ``high`` are the enclosure endpoints as exact rationals, and
    both round to ``value``. For the exact path they are equal and the value
    is the rounding of a number known exactly.
    """

    value: float
    dtype: str
    method: str
    low: Fraction
    high: Fraction
    precision: int | None
    validated: bool = True

    def describe(self) -> str:
        """A line of diagnostics for a failed comparison."""
        if self.method == "exact rational":
            return f"exact rational, rounded once to {self.dtype}"
        width = self.high - self.low
        return (
            f"enclosure at {self.precision} digits, "
            f"width {float(width):.3e}, both ends round to {self.value!r}"
        )


def _as_fraction(value: float) -> Fraction:
    """A binary float is exactly a rational; this is that rational."""
    return Fraction(*value.as_integer_ratio())


def _at_least(numerator: int, denominator: int, exponent: int) -> bool:
    """Whether ``numerator / denominator >= 2 ** exponent``, decided exactly."""
    if exponent >= 0:
        return numerator >= denominator << exponent
    return numerator << -exponent >= denominator


def round_fraction(value: Fraction, dtype: str) -> float:
    """Round an exact rational into a format, ties to even (section 5.2).

    The rounding is performed once, directly from the rational, using integer
    arithmetic alone. Nothing passes through an intermediate format, so a
    ``float32`` result is not rounded through ``float64``.
    """
    if value == 0:
        return 0.0
    parameters = FORMATS[dtype]
    precision = parameters["precision"]
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    numerator, denominator = magnitude.numerator, magnitude.denominator

    # The leading bit's exponent: the largest e with 2**e <= magnitude. The
    # bit lengths put it within one, and the loops settle which.
    leading = numerator.bit_length() - denominator.bit_length()
    while not _at_least(numerator, denominator, leading):
        leading -= 1
    while _at_least(numerator, denominator, leading + 1):
        leading += 1

    # The result's quantum, floored at the subnormal quantum.
    quantum = max(leading - precision + 1, parameters["min_exponent"] - precision + 1)
    if quantum >= 0:
        denominator <<= quantum
    else:
        numerator <<= -quantum
    units, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and units % 2):
        units += 1

    # Overflow, decided on the rounded value: units * 2**quantum > max finite.
    largest = (2**precision - 1) * 2 ** (parameters["max_exponent"] - precision + 1)
    if _at_least(units, 1, 0) and Fraction(units, 1) * Fraction(2) ** quantum > largest:
        return sign * math.inf
    return sign * math.ldexp(float(units), quantum)


def _decimal_interval(quantity: Decimal, digits: int) -> tuple[Fraction, Fraction]:
    """Widen a correctly rounded decimal into an interval containing the truth.

    ``ln`` and ``exp`` are documented as correctly rounded, so the true value
    is within half a unit in the last place. One whole unit is used, which is
    a valid — and deliberately unsubtle — bound either way.
    """
    exact = Fraction(quantity)
    if quantity == 0:
        # Only reachable for ln(1), which the callers handle before this.
        unit = Fraction(1, 10**digits)
    else:
        unit = Fraction(10) ** (quantity.adjusted() - digits + 1)
    return exact - unit, exact + unit


def _to_decimal(value: Fraction, digits: int, rounding: str) -> Decimal:
    """Convert a rational to a decimal in a stated direction.

    Directed rounding keeps an interval endpoint on the outside of the true
    value, so the conversion can only widen the enclosure, never narrow it.
    """
    with localcontext() as context:
        context.prec = digits
        context.rounding = rounding
        context.Emax = 10**9
        context.Emin = -(10**9)
        return Decimal(value.numerator) / Decimal(value.denominator)


def _exact_integral_power(base: float, exponent: int, dtype: str) -> Reference | None:
    """``x ** n`` as an exact rational, when its size is manageable."""
    rational = _as_fraction(base)
    if rational == 0:
        raise OutsideReferenceDomain("a zero base is a special value")
    size = max(
        rational.numerator.bit_length(), rational.denominator.bit_length()
    ) * abs(exponent)
    if size > EXACT_BIT_LIMIT:
        return None
    exact = rational**exponent
    return Reference(
        value=round_fraction(exact, dtype),
        dtype=dtype,
        method="exact rational",
        low=exact,
        high=exact,
        precision=None,
    )


def _enclosure(base: float, exponent: float, dtype: str) -> Reference:
    """``exp(y ln x)`` as an interval, refined until the rounding is decided."""
    magnitude = abs(base)
    negative = base < 0.0
    if negative and not float(exponent).is_integer():
        raise OutsideReferenceDomain("a negative base needs an integral exponent")
    sign = -1 if (negative and int(exponent) % 2) else 1
    exponent_exact = _as_fraction(float(exponent))

    for digits in PRECISIONS:
        with localcontext() as context:
            context.prec = digits
            context.Emax = 10**9
            context.Emin = -(10**9)
            logarithm = Decimal(magnitude).ln()
        low, high = _decimal_interval(logarithm, digits)

        # The product interval; multiplying by an exact rational adds nothing.
        ends = (exponent_exact * low, exponent_exact * high)
        product_low, product_high = min(ends), max(ends)

        if product_low > CERTAIN_OVERFLOW:
            value = sign * math.inf
            return Reference(
                value, dtype, "certain overflow", product_low, product_high, digits
            )
        if product_high < CERTAIN_UNDERFLOW:
            value = math.copysign(0.0, sign)
            return Reference(
                value, dtype, "certain underflow", product_low, product_high, digits
            )

        with localcontext() as context:
            context.prec = digits
            context.Emax = 10**9
            context.Emin = -(10**9)
            lower = _to_decimal(product_low, digits, "ROUND_FLOOR").exp()
            upper = _to_decimal(product_high, digits, "ROUND_CEILING").exp()
        result_low = _decimal_interval(lower, digits)[0]
        result_high = _decimal_interval(upper, digits)[1]

        rounded_low = round_fraction(sign * result_low, dtype)
        rounded_high = round_fraction(sign * result_high, dtype)
        if rounded_low == rounded_high and math.copysign(
            1.0, rounded_low
        ) == math.copysign(1.0, rounded_high):
            return Reference(
                value=rounded_low,
                dtype=dtype,
                method="enclosure",
                low=sign * result_low if sign > 0 else sign * result_high,
                high=sign * result_high if sign > 0 else sign * result_low,
                precision=digits,
            )

    raise UnresolvedReference(
        f"{base!r} ** {exponent!r} in {dtype} did not resolve at "
        f"{PRECISIONS[-1]} digits: the enclosure still spans "
        f"{rounded_low!r} to {rounded_high!r}"
    )


def reference_power(base: float, exponent: float, dtype: str) -> Reference:
    """The correctly rounded ``base ** exponent`` in ``dtype``.

    Raises :class:`OutsideReferenceDomain` for the operand pairs section
    12.6.1 governs exactly — zeros, infinities and NaNs — which the ULP metric
    does not apply to, and :class:`UnresolvedReference` when the rounding
    cannot be decided.
    """
    if dtype not in FORMATS:
        raise ValueError(f"unsupported format {dtype!r}")
    if base != base or exponent != exponent:
        raise OutsideReferenceDomain("NaN is compared by classification")
    if math.isinf(base) or math.isinf(exponent):
        raise OutsideReferenceDomain("an infinite operand is a special value")
    if base == 0.0 or exponent == 0.0:
        raise OutsideReferenceDomain("a zero operand is a special value")
    if base < 0.0 and not float(exponent).is_integer():
        raise OutsideReferenceDomain("no real result; section 12.3.3 gives NaN")

    if float(exponent).is_integer() and abs(exponent) < 1 << 30:
        exact = _exact_integral_power(base, int(exponent), dtype)
        if exact is not None:
            return exact
    return _enclosure(base, exponent, dtype)


__all__ = [
    "FORMATS",
    "OutsideReferenceDomain",
    "Reference",
    "UnresolvedReference",
    "reference_power",
    "round_fraction",
]
