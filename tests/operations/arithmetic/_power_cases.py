"""Deterministic operand pairs for the D5 accuracy measurements.

Constructed cases come first and carry the weight; the random supplement is
additional coverage, never a substitute for a boundary that was reasoned
about. Every value is fixed, so a failure is reproducible from its operands
alone.

The pairs here are *logical*: each is rounded into the dtype under test before
use, and the accuracy harness reads the operands back out of the tensor so
that the reference is asked about exactly the values the kernel saw.
"""

from __future__ import annotations

import math
import random
import struct

#: Section 3.2 boundaries, written out.
MAX = {"float32": 3.4028234663852886e38, "float64": 1.7976931348623157e308}
MIN_NORMAL = {"float32": 1.1754943508222875e-38, "float64": 2.2250738585072014e-308}
SMALLEST = {"float32": 1.401298464324817e-45, "float64": 5e-324}


def _round(value: float, dtype: str) -> float:
    """The literal as the dtype holds it, so a case states its own operands."""
    code = "<f" if dtype == "float32" else "<d"
    try:
        return struct.unpack(code, struct.pack(code, value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def constructed(dtype: str) -> list[tuple[str, float, float]]:
    """The reasoned cases, labelled by what each one is about."""
    cases: list[tuple[str, float, float]] = []
    add = cases.append

    # Ordinary positive bases with fractional exponents.
    for base in (0.5, 0.75, 1.5, 2.0, 2.5, 3.0, 7.0, 10.0, 97.0, 1234.5):
        for exponent in (0.5, 0.25, 1.5, 2.5, 0.3333333333333333, -0.5, -1.5, 3.7):
            add(("ordinary", base, exponent))

    # Negative bases with integral exponents, both parities, both signs.
    for base in (-1.5, -2.0, -3.0, -7.5, -0.25):
        for exponent in (2.0, 3.0, 4.0, 5.0, -2.0, -3.0, 17.0, -17.0):
            add(("negative base, integral exponent", base, exponent))

    # Bases close to one, where ln x is tiny and the exponent amplifies it.
    ulp = 2.0**-23 if dtype == "float32" else 2.0**-52
    for offset in (ulp, 2 * ulp, 16 * ulp, -ulp, -2 * ulp, -16 * ulp):
        for exponent in (1.5, 100.0, 1e4, 1e6, -1e4, 0.5, -0.5):
            add(("base near one", 1.0 + offset, exponent))

    # Large and small magnitudes.
    for base in (1e10, 1e-10, 1e30, 1e-30, 1e100, 1e-100):
        if dtype == "float32" and (base >= 1e38 or base <= 1e-38):
            continue
        for exponent in (0.5, 1.5, 2.0, -0.5, -1.5, 0.1, 3.3):
            add(("large or small magnitude", base, exponent))

    # Near overflow: the result sits just under, at, and just over the top.
    top = MAX[dtype]
    for exponent in (1.0, 1.0000001, 0.9999999, 2.0, 0.5):
        add(("near overflow", top, exponent))
        add(("near overflow", _round(top * 0.999, dtype), exponent))
    add(("just over the top", top, 1.001))
    add(("just under the top", top, 0.999))

    # Near underflow and into the subnormals.
    smallest = SMALLEST[dtype]
    normal = MIN_NORMAL[dtype]
    for exponent in (1.0, 1.0000001, 0.9999999, 0.5, 2.0, 1.5):
        add(("subnormal operand", smallest, exponent))
        add(("subnormal operand", 4 * smallest, exponent))
        add(("normal boundary", normal, exponent))
        add(("just below normal", _round(normal * 0.5, dtype), exponent))
    # Results that land in the subnormal range.
    for base, exponent in (
        (1e-20, 2.0) if dtype == "float32" else (1e-160, 2.0),
        (1e-22, 2.0) if dtype == "float32" else (1e-170, 2.0),
        (2e-19, 2.0) if dtype == "float32" else (2e-155, 2.0),
    ):
        add(("subnormal result", base, exponent))
    for base, exponent in (
        ((1e-23, 2.0), (-1e-23, 3.0))
        if dtype == "float32"
        else ((1e-180, 2.0), (-1e-180, 3.0))
    ):
        add(("underflow to zero", base, exponent))

    # Exactly representable results, where the correct answer is known without
    # any high-precision machinery at all.
    for base, exponent in (
        (2.0, 10.0),
        (2.0, -10.0),
        (4.0, 0.5),
        (9.0, 0.5),
        (16.0, 0.25),
        (2.0, 24.0),
        (0.5, 30.0),
        (-2.0, 7.0),
        (-2.0, 8.0),
        (1.5, 2.0),
        (2.5, 2.0),
        (1024.0, 0.5),
        (0.0625, -1.0),
    ):
        add(("exactly representable", base, exponent))

    # Either side of a rounding midpoint: bases whose square lands close to
    # one. 1 + 2**-12 squared is an exact float32 midpoint; its binary64
    # counterpart is 1 + 2**-27.
    half = 2.0**-12 if dtype == "float32" else 2.0**-27
    for offset in (
        half,
        _round(math.nextafter(half, 0.0), dtype),
        _round(math.nextafter(half, 1.0), dtype),
    ):
        add(("rounding midpoint", 1.0 + offset, 2.0))

    return [
        (label, _round(base, dtype), _round(exponent, dtype))
        for label, base, exponent in cases
    ]


#: Literals that binary32 cannot hold exactly. Deliberately *not* passed
#: through :func:`_round`: the point of these is that rule S3 changes them,
#: so the scalar path is measured against the converted value rather than the
#: literal. The first is the case reported against the accuracy harness.
INEXACT_SCALARS: tuple[float, ...] = (
    1.0000001,
    0.9999999,
    1.1,
    0.1,
    3.3,
    2.718281828459045,
    1.0000000596046448,
    -1.0000001,
)


def inexact_scalars(dtype: str) -> list[tuple[str, float, float]]:
    """Pairs whose *scalar* operand the conversion actually changes.

    The tensor operand is rounded into the dtype as usual; the scalar is left
    as written, so the public operation performs a real conversion. For
    ``float64`` the conversion is the identity — binary64 is the literal's own
    format — and these pairs then simply add ordinary coverage.

    A negative base is paired only with an integral exponent, since section
    12.3.3 gives NaN otherwise and the ULP metric does not apply there.
    """
    bases = (1.1754943508222875e-38, 1.5, 0.75, 97.0, 2.0, 1e-20, -2.0, -1.5)
    cases: list[tuple[str, float, float]] = []
    for base in bases:
        for scalar in INEXACT_SCALARS:
            if base < 0.0:
                continue
            cases.append(("inexact scalar", _round(base, dtype), scalar))
    # The reflected form takes the scalar as the *base*, so it is listed with
    # the scalar first; the harness reads whichever operand is the scalar.
    #
    # Large exponents are included deliberately. exp amplifies a change in
    # y ln x by |y|, so a one-ULP difference in a base near one moves the
    # result by one ULP at an exponent of 1.5 but by thousands at 10**4. With
    # small exponents alone, recording the literal instead of its conversion
    # would stay inside the bound and the measurement would not see it.
    for scalar in INEXACT_SCALARS:
        exponents = (1.5, 2.0, -0.5, 3.0, 0.25)
        if abs(abs(scalar) - 1.0) < 0.01:
            exponents += (100.0, 1e4, -1e4)
        for exponent in exponents:
            if scalar < 0.0 and not float(exponent).is_integer():
                continue
            cases.append(("inexact scalar", scalar, _round(exponent, dtype)))
    return cases


def sampled(
    dtype: str, count: int = 400, seed: int = 20260918
) -> list[tuple[str, float, float]]:
    """A deterministic random supplement, spread over the exponent range."""
    generator = random.Random(seed)
    limit = 37 if dtype == "float32" else 300
    cases = []
    for _ in range(count):
        base = generator.uniform(-1.0, 1.0) * 10 ** generator.uniform(-limit, limit)
        exponent = generator.uniform(-1.0, 1.0) * 10 ** generator.uniform(-2, 2)
        if base < 0.0:
            exponent = float(round(exponent))  # a real result needs an integer
            if exponent == 0.0:
                exponent = 3.0
        if base == 0.0 or exponent == 0.0:
            continue
        cases.append(("sampled", _round(base, dtype), _round(exponent, dtype)))
    return cases


def all_cases(dtype: str) -> list[tuple[str, float, float]]:
    """Every case for a dtype, constructed first."""
    return constructed(dtype) + sampled(dtype)


__all__ = [
    "INEXACT_SCALARS",
    "MAX",
    "MIN_NORMAL",
    "SMALLEST",
    "all_cases",
    "constructed",
    "inexact_scalars",
    "sampled",
]
