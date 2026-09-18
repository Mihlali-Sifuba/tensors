"""The arithmetic specification, expressed independently of the package.

`docs/arithmetic-semantics.md` is the oracle for these tests, so the rules it
states are written out here from the document rather than obtained by running
any backend. Nothing in this module imports ``tensors``: an expectation that
came from the implementation would only confirm that the implementation agrees
with itself.

Each function names the section it comes from.
"""

from __future__ import annotations

import math

#: Section 3.1 — the public integer dtypes, their width, and their range.
INTEGER_DTYPES: dict[str, tuple[int, bool]] = {
    "uint8": (8, False),
    "int8": (8, True),
    "int16": (16, True),
    "int32": (32, True),
    "int64": (64, True),
}

#: Section 3.2 — precision of the public floating dtypes.
FLOAT_PRECISION: dict[str, int] = {"float32": 24, "float64": 53}

#: Section 3.2 — the largest finite value of each floating dtype, as an exact
#: integer, so representability questions need no floating arithmetic.
FLOAT_MAX: dict[str, int] = {
    "float32": (2**24 - 1) * 2 ** (127 - 23),
    "float64": (2**53 - 1) * 2 ** (1023 - 52),
}

DTYPES: tuple[str, ...] = (
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
)

FLOAT_DTYPES: tuple[str, ...] = ("float32", "float64")

#: Section 6.2 — the sentinel for a combination that requires an explicit cast.
CAST = "cast"


def integer_range(dtype: str) -> tuple[int, int]:
    """Return the inclusive range of an integer dtype (section 3.1)."""
    width, signed = INTEGER_DTYPES[dtype]
    if signed:
        return -(2 ** (width - 1)), 2 ** (width - 1) - 1
    return 0, 2**width - 1


def wrap(result: int, dtype: str) -> int:
    """Apply the fixed-width wraparound rule of section 4.2.

    Signed:   ((r + 2**(w-1)) mod 2**w) - 2**(w-1)
    Unsigned: r mod 2**w
    """
    width, signed = INTEGER_DTYPES[dtype]
    if not signed:
        return result % 2**width
    return ((result + 2 ** (width - 1)) % 2**width) - 2 ** (width - 1)


def exactly_representable(value: int, dtype: str) -> bool:
    """Whether one integer is exact in a floating dtype (section 6.5, S4).

    Writing |n| = m * 2**k with m odd, n is exact when m needs at most p bits
    and |n| is within the format's finite range. The interval [-2**p, 2**p] is
    sufficient but not necessary, which is the point of the corrected rule.
    """
    if value == 0:
        return True
    odd = abs(value)
    while odd % 2 == 0:
        odd //= 2
    return odd.bit_length() <= FLOAT_PRECISION[dtype] and abs(value) <= FLOAT_MAX[dtype]


def dtype_fits_in_float(integer_dtype: str, float_dtype: str) -> bool:
    """Whether *every* value of an integer dtype is exact (section 3.3).

    This is the range question promotion asks, and it is deliberately not the
    per-value question :func:`exactly_representable` answers.
    """
    lower, upper = integer_range(integer_dtype)
    limit = 2 ** FLOAT_PRECISION[float_dtype]
    return -limit <= lower and upper <= limit


def promote(left: str, right: str) -> str:
    """Return the result dtype for ``+``, ``-`` and ``*`` (section 6.2).

    Derived from the principles P-a to P-e rather than copied from the table,
    so that the table in the document and the behaviour of the package are
    checked against the same independent derivation.
    """
    if left == right:
        return left
    both_integer = left in INTEGER_DTYPES and right in INTEGER_DTYPES
    if both_integer:
        low = min(integer_range(left)[0], integer_range(right)[0])
        high = max(integer_range(left)[1], integer_range(right)[1])
        for candidate in INTEGER_DTYPES:
            candidate_low, candidate_high = integer_range(candidate)
            if candidate_low <= low and high <= candidate_high:
                return candidate
        return CAST
    if left not in INTEGER_DTYPES and right not in INTEGER_DTYPES:
        return "float64"
    integer = left if left in INTEGER_DTYPES else right
    floating = right if left in INTEGER_DTYPES else left
    for candidate in FLOAT_DTYPES:
        if FLOAT_PRECISION[candidate] < FLOAT_PRECISION[floating]:
            continue  # never narrow the floating operand
        if dtype_fits_in_float(integer, candidate):
            return candidate
    return CAST


def divide_result(left: str, right: str) -> str:
    """Return the result dtype of true division (section 7, rule D).

    Division never yields an integer. Integer operands promote first, and the
    promoted integer dtype must then fit exactly in a floating dtype.
    """
    promoted = promote(left, right)
    if promoted == CAST:
        return CAST
    if promoted not in INTEGER_DTYPES:
        return promoted
    for candidate in FLOAT_DTYPES:
        if dtype_fits_in_float(promoted, candidate):
            return candidate
    return CAST


def scalar_rule(dtype: str, value: object) -> tuple[str, bool]:
    """Return ``(rule, converts)`` for a Python scalar (section 6.5).

    S1  int with an integer tensor  — in range, exactly
    S2  float with an integer tensor — integral and in range
    S3  float with a floating tensor — rounded; must not round to infinity
    S4  int with a floating tensor  — that integer exactly representable
    """
    if isinstance(value, bool):
        return "bool", False
    integer_target = dtype in INTEGER_DTYPES
    if isinstance(value, int):
        if integer_target:
            low, high = integer_range(dtype)
            return "S1", low <= value <= high
        return "S4", exactly_representable(value, dtype)
    if isinstance(value, float):
        if integer_target:
            if not math.isfinite(value) or not value.is_integer():
                return "S2", False
            low, high = integer_range(dtype)
            return "S2", low <= int(value) <= high
        if math.isnan(value) or math.isinf(value):
            return "S3", True
        return "S3", rounds_to_finite(value, dtype)
    return "unsupported", False


#: Exponent of the largest finite value in each format.
MAX_EXPONENT: dict[str, int] = {"float32": 127, "float64": 1023}


def overflow_threshold(dtype: str) -> int | float:
    """The magnitude at which round-to-nearest delivers infinity.

    A finite value rounds to the largest finite value while it stays below the
    midpoint between that value and the first unrepresentable power of two;
    at or above the midpoint it rounds to infinity. Returned as an exact
    rational scaled by two so no floating arithmetic enters the comparison.
    """
    return FLOAT_MAX[dtype] + 2 ** (MAX_EXPONENT[dtype] + 1)


#: Exponent of the smallest normal value in each format, which fixes the
#: quantum subnormal results are rounded to.
MIN_EXPONENT: dict[str, int] = {"float32": -126, "float64": -1022}


def round_to_format(value: float, dtype: str) -> float:
    """Round a binary64 value to a floating dtype (section 6.5, S3).

    Round-to-nearest, ties-to-even, derived from the format's precision and
    exponent range with exact integer arithmetic. Nothing here consults a
    conversion performed by the package, by ``array`` or by ``struct``, so it
    is an independent statement of what S3 requires.

    NaN, the infinities and the signed zeros are returned unchanged. A value
    that rounds past the largest finite value returns an infinity; S3 refuses
    such a literal separately, which :func:`rounds_to_finite` decides.
    """
    if math.isnan(value) or math.isinf(value) or value == 0.0:
        return value
    precision = FLOAT_PRECISION[dtype]
    sign = -1.0 if value < 0.0 else 1.0
    numerator, denominator = abs(value).as_integer_ratio()
    # frexp is exact: 2**leading <= |value| < 2**(leading + 1).
    leading = math.frexp(abs(value))[1] - 1
    # The quantum of the result, floored at the subnormal quantum so that
    # gradual underflow rounds to the same grid the format actually has.
    quantum = max(leading - precision + 1, MIN_EXPONENT[dtype] - precision + 1)
    if quantum >= 0:
        denominator <<= quantum
    else:
        numerator <<= -quantum
    units, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and units % 2):
        units += 1  # ties to even leaves an even unit count alone
    magnitude = math.ldexp(float(units), quantum)
    if magnitude > FLOAT_MAX[dtype]:
        return sign * math.inf
    return sign * magnitude


def rounds_to_finite(value: float, dtype: str) -> bool:
    """Whether a finite literal survives conversion (section 6.5, S3)."""
    if math.isnan(value) or math.isinf(value):
        return True
    # Compare exactly: 2*|v| against max + 2**(emax+1), the doubled midpoint.
    numerator, denominator = abs(value).as_integer_ratio()
    return 2 * numerator < overflow_threshold(dtype) * denominator


__all__ = [
    "CAST",
    "DTYPES",
    "FLOAT_DTYPES",
    "FLOAT_MAX",
    "FLOAT_PRECISION",
    "MAX_EXPONENT",
    "MIN_EXPONENT",
    "INTEGER_DTYPES",
    "divide_result",
    "dtype_fits_in_float",
    "exactly_representable",
    "integer_range",
    "overflow_threshold",
    "promote",
    "round_to_format",
    "rounds_to_finite",
    "scalar_rule",
    "wrap",
]
