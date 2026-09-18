"""The ULP metric and the classification rules around it (section 12.6.4).

The metric applies to finite, non-zero results of the same sign and to nothing
else. Zeros, infinities, NaNs and sign disagreements are *classifications*, not
distances, and this module never reports an integer distance for one of them —
a signed-zero mismatch is a failure, not a large number.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

#: Section 12.6.2 — the bound for each declared result dtype, in ULP.
BOUNDS: dict[str, int] = {"float64": 2, "float32": 4}

_FORMAT: dict[str, tuple[str, str, int]] = {
    "float32": ("<f", "<i", 32),
    "float64": ("<d", "<q", 64),
}


def monotone(value: float, dtype: str) -> int:
    """Map a finite value to an integer on which neighbours differ by one.

    Interpret the bit pattern as a signed integer ``i``; where ``i`` is
    negative, replace it with ``INT_MIN - i``. Adjacent representable values
    then differ by exactly one, the subnormal-to-normal boundary included.
    """
    pack, unpack, width = _FORMAT[dtype]
    ordered = struct.unpack(unpack, struct.pack(pack, value))[0]
    if ordered < 0:
        return (-(1 << (width - 1))) - ordered
    return ordered


@dataclass(frozen=True)
class Comparison:
    """The outcome of one accuracy comparison, with its diagnostics."""

    conforms: bool
    #: The ULP distance, or ``None`` where the metric does not apply.
    distance: int | None
    #: Why the metric did not apply, or ``""`` when it did.
    classification: str

    def __bool__(self) -> bool:
        return self.conforms


def _class_of(value: float) -> str:
    if value != value:
        return "NaN"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if value == 0.0:
        return "+0" if math.copysign(1.0, value) > 0 else "-0"
    return "finite"


def compare(produced: float, expected: float, dtype: str) -> Comparison:
    """Assess one result against the reference under section 12.6.4."""
    bound = BOUNDS[dtype]
    produced_class = _class_of(produced)
    expected_class = _class_of(expected)

    if expected_class == "NaN" or produced_class == "NaN":
        # NaN is compared by classification only; payload and sign are
        # unspecified and are never tested.
        conforms = produced_class == "NaN" == expected_class
        return Comparison(conforms, None, "NaN classification")
    if expected_class != "finite" or produced_class != "finite":
        # A zero or an infinity is exact on both sides, sign included, and a
        # finite/non-finite disagreement is a failure rather than a distance.
        return Comparison(
            produced_class == expected_class, None, "exact class comparison"
        )
    if (produced > 0.0) != (expected > 0.0):
        return Comparison(False, None, "opposite signs")

    distance = abs(monotone(produced, dtype) - monotone(expected, dtype))
    return Comparison(distance <= bound, distance, "")


def describe(
    *,
    base,
    exponent,
    dtype: str,
    backend: str,
    mode: str,
    produced: float,
    expected: float,
    comparison: Comparison,
    reference: str = "",
) -> str:
    """A failure diagnostic carrying everything needed to reproduce the case."""
    lines = [
        f"{backend}/{mode}/{dtype}: {base!r} ** {exponent!r}",
        f"  expected {expected!r} ({_class_of(expected)})",
        f"  produced {produced!r} ({_class_of(produced)})",
    ]
    if comparison.distance is None:
        lines.append(f"  no ULP distance applies: {comparison.classification}")
    else:
        lines.append(f"  {comparison.distance} ULP, bound {BOUNDS[dtype]} ULP")
    if reference:
        lines.append(f"  reference: {reference}")
    return "\n".join(lines)


__all__ = ["BOUNDS", "Comparison", "compare", "describe", "monotone"]
