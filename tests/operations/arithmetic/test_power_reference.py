"""The high-precision reference is validated before it judges any kernel.

`docs/arithmetic-semantics.md` section 12.6.5. Nothing here compares against
``math.pow``, NumPy or CuPy: a library's answer cannot validate the reference
that is meant to measure it.

What can validate it is arithmetic that is exact. The evidence below is of
three kinds:

* results known exactly without any high-precision machinery — small powers,
  powers of two, exact square roots — compared against a rounding derived
  independently of the reference;
* the enclosure procedure checked against the exact rational procedure, both
  by requiring the enclosure to *bracket* a value known exactly and by
  requiring the two procedures to agree on the value they return;
* the refusal to answer: an operand pair whose true result is exactly a
  rounding midpoint can never be resolved by an enclosure, and the reference
  must report it unresolved rather than guess.
"""

from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction

from . import _reference
from ._reference import (
    OutsideReferenceDomain,
    UnresolvedReference,
    reference_power,
    round_fraction,
)
from ._support import float_bits


class RoundingARational(unittest.TestCase):
    """``round_fraction`` is the one place a result reaches a format."""

    def test_exact_values_of_both_formats(self):
        for dtype_name in ("float32", "float64"):
            for value in (1.0, 1.5, -2.25, 1024.0, 0.125, 2.0**-100, -(2.0**80)):
                with self.subTest(dtype=dtype_name, value=value):
                    exact = Fraction(*value.as_integer_ratio())
                    self.assertEqual(round_fraction(exact, dtype_name), value)

    def test_ties_go_to_even(self):
        """Midpoints are constructed exactly, so the rule is what is tested."""
        for dtype_name, unit in (("float32", 2.0**-23), ("float64", 2.0**-52)):
            one = Fraction(1)
            half = Fraction(*(unit / 2).as_integer_ratio())
            with self.subTest(dtype=dtype_name):
                # 1 has an even significand, so the midpoint above it ties down.
                self.assertEqual(round_fraction(one + half, dtype_name), 1.0)
                # The next value up has an odd significand, so its upper
                # midpoint ties up, away from it.
                step = Fraction(*unit.as_integer_ratio())
                self.assertEqual(
                    round_fraction(one + step + half, dtype_name), 1.0 + 2 * unit
                )

    def test_either_side_of_a_tie(self):
        for dtype_name, unit in (("float32", 2.0**-23), ("float64", 2.0**-52)):
            half = Fraction(*(unit / 2).as_integer_ratio())
            nudge = Fraction(1, 2**200)
            with self.subTest(dtype=dtype_name):
                self.assertEqual(
                    round_fraction(Fraction(1) + half - nudge, dtype_name), 1.0
                )
                self.assertEqual(
                    round_fraction(Fraction(1) + half + nudge, dtype_name), 1.0 + unit
                )

    def test_the_subnormal_grid(self):
        smallest = Fraction(1, 2**149)
        self.assertEqual(round_fraction(smallest, "float32"), 1.401298464324817e-45)
        self.assertEqual(
            round_fraction(3 * smallest, "float32"), 3 * 1.401298464324817e-45
        )
        # Half a quantum ties to even, which is zero.
        self.assertEqual(round_fraction(smallest / 2, "float32"), 0.0)
        # Three halves of a quantum ties up, to two quanta.
        self.assertEqual(
            round_fraction(3 * smallest / 2, "float32"), 2 * 1.401298464324817e-45
        )

    def test_overflow_reaches_infinity(self):
        for dtype_name, largest in (
            ("float32", 3.4028234663852886e38),
            ("float64", 1.7976931348623157e308),
        ):
            with self.subTest(dtype=dtype_name):
                exact = Fraction(*largest.as_integer_ratio())
                self.assertEqual(round_fraction(exact, dtype_name), largest)
                self.assertTrue(math.isinf(round_fraction(exact * 4, dtype_name)))
                self.assertTrue(math.isinf(round_fraction(-exact * 4, dtype_name)))

    def test_a_binary32_result_is_not_rounded_through_binary64(self):
        """Constructed so that single and double rounding visibly differ.

        The value sits one part in ``2**80`` above an exact ``float32``
        midpoint. Rounded once it goes up; rounded through ``float64`` it
        first lands exactly on the midpoint, which then ties *down* to the
        even neighbour. Section 12.6.5 step 3 requires the former.
        """
        midpoint = Fraction(1) + Fraction(*(2.0**-24).as_integer_ratio())
        value = midpoint + Fraction(1, 2**80)

        once = round_fraction(value, "float32")
        through_binary64 = round_fraction(
            Fraction(*round_fraction(value, "float64").as_integer_ratio()), "float32"
        )

        self.assertEqual(once, 1.0 + 2.0**-23)
        self.assertEqual(through_binary64, 1.0)
        self.assertNotEqual(
            float_bits(once, "float32"), float_bits(through_binary64, "float32")
        )


class ExactlyKnownResults(unittest.TestCase):
    """Cases whose correct answer needs no high-precision machinery."""

    #: (base, exponent, the exact result as a rational)
    EXACT = (
        (2.0, 10.0, Fraction(1024)),
        (2.0, -10.0, Fraction(1, 1024)),
        (4.0, 0.5, Fraction(2)),
        (9.0, 0.5, Fraction(3)),
        (16.0, 0.25, Fraction(2)),
        (1.5, 2.0, Fraction(9, 4)),
        (2.5, 3.0, Fraction(125, 8)),
        (0.5, 30.0, Fraction(1, 2**30)),
        (-2.0, 7.0, Fraction(-128)),
        (-2.0, 8.0, Fraction(256)),
        (-1.5, 3.0, Fraction(-27, 8)),
        (-0.25, -3.0, Fraction(-64)),
        (1024.0, 0.5, Fraction(32)),
        (3.0, 40.0, Fraction(3) ** 40),
    )

    def test_the_reference_agrees_with_the_exact_rational(self):
        for base, exponent, exact in self.EXACT:
            for dtype_name in ("float32", "float64"):
                with self.subTest(base=base, exponent=exponent, dtype=dtype_name):
                    produced = reference_power(base, exponent, dtype_name)
                    self.assertEqual(
                        float_bits(produced.value, dtype_name),
                        float_bits(round_fraction(exact, dtype_name), dtype_name),
                        f"{base!r} ** {exponent!r}: {produced.describe()}",
                    )

    def test_a_negative_base_keeps_the_parity_sign(self):
        for base, exponent, exact in self.EXACT:
            if base >= 0:
                continue
            for dtype_name in ("float32", "float64"):
                with self.subTest(base=base, exponent=exponent, dtype=dtype_name):
                    produced = reference_power(base, exponent, dtype_name).value
                    self.assertEqual(
                        math.copysign(1.0, produced),
                        1.0 if exact > 0 else -1.0,
                        f"{base!r} ** {exponent!r} has the wrong sign",
                    )


class TheEnclosureIsSound(unittest.TestCase):
    """The interval procedure, checked against arithmetic that is exact."""

    #: Integral exponents, so the true value is known as a rational and the
    #: enclosure can be required to contain it.
    BRACKETED = (
        (1.5, 3),
        (2.5, 7),
        (0.7, 11),
        (3.0, 20),
        (1.0009765625, 5),
        (0.5, 100),
        (1.1, 17),
        (7.0, 9),
        (0.001, 5),
        (100000.0, 6),
        (-2.5, 4),
        (-3.0, 5),
        (1.0000001, 3),
        (0.9999999, 9),
    )

    def enclose(self, base, exponent, dtype_name):
        """Force the enclosure path, whatever the exact path would do."""
        saved = _reference.EXACT_BIT_LIMIT
        _reference.EXACT_BIT_LIMIT = 0
        try:
            return _reference.reference_power(base, float(exponent), dtype_name)
        finally:
            _reference.EXACT_BIT_LIMIT = saved

    def test_the_enclosure_contains_the_exact_value(self):
        """The interval must bracket a number known exactly, or it is wrong."""
        for base, exponent in self.BRACKETED:
            exact = Fraction(*float(base).as_integer_ratio()) ** exponent
            for dtype_name in ("float32", "float64"):
                with self.subTest(base=base, exponent=exponent, dtype=dtype_name):
                    produced = self.enclose(base, exponent, dtype_name)
                    self.assertEqual(produced.method, "enclosure")
                    self.assertLessEqual(
                        produced.low, exact, f"{base!r} ** {exponent}: low end too high"
                    )
                    self.assertLessEqual(
                        exact,
                        produced.high,
                        f"{base!r} ** {exponent}: high end too low",
                    )

    def test_the_two_procedures_return_the_same_value(self):
        for base, exponent in self.BRACKETED:
            for dtype_name in ("float32", "float64"):
                with self.subTest(base=base, exponent=exponent, dtype=dtype_name):
                    exact = reference_power(base, float(exponent), dtype_name)
                    enclosed = self.enclose(base, exponent, dtype_name)
                    self.assertEqual(exact.method, "exact rational")
                    self.assertEqual(
                        float_bits(exact.value, dtype_name),
                        float_bits(enclosed.value, dtype_name),
                        f"{base!r} ** {exponent}: {exact.describe()} against "
                        f"{enclosed.describe()}",
                    )

    def test_both_endpoints_round_to_the_returned_value(self):
        """The acceptance condition, asserted rather than assumed."""
        for base, exponent in (
            (2.0, 0.5),
            (3.0, 0.3333333333333333),
            (1.5, 1.5),
            (10.0, 0.1),
            (0.25, -0.75),
            (7.5, 2.5),
        ):
            for dtype_name in ("float32", "float64"):
                with self.subTest(base=base, exponent=exponent, dtype=dtype_name):
                    produced = reference_power(base, exponent, dtype_name)
                    self.assertEqual(produced.method, "enclosure")
                    self.assertLessEqual(produced.low, produced.high)
                    for end in (produced.low, produced.high):
                        self.assertEqual(
                            float_bits(round_fraction(end, dtype_name), dtype_name),
                            float_bits(produced.value, dtype_name),
                        )

    def test_the_working_precision_is_recorded(self):
        produced = reference_power(2.0, 0.5, "float64")
        self.assertIsNotNone(produced.precision)
        self.assertIn(produced.precision, _reference.PRECISIONS)
        self.assertIn("enclosure at", produced.describe())


class ItRefusesToGuess(unittest.TestCase):
    """Section 12.6.5 step 5: never return an unvalidated reference."""

    #: ``(1 + 2**-12) ** 2`` is ``1 + 2**-11 + 2**-24`` exactly, and ``2**-24``
    #: is exactly half a ``float32`` ULP at one. No enclosure of finite width
    #: can decide which side of that midpoint the true value falls, because it
    #: falls on it.
    MIDPOINT_BASE = 1.0 + 2.0**-12

    def test_the_exact_path_resolves_the_midpoint_by_ties_to_even(self):
        produced = reference_power(self.MIDPOINT_BASE, 2.0, "float32")
        self.assertEqual(produced.method, "exact rational")
        self.assertEqual(produced.value, 1.0 + 2.0**-11)

    def test_the_enclosure_path_reports_it_unresolved(self):
        saved = _reference.EXACT_BIT_LIMIT
        _reference.EXACT_BIT_LIMIT = 0
        try:
            with self.assertRaises(UnresolvedReference):
                _reference.reference_power(self.MIDPOINT_BASE, 2.0, "float32")
        finally:
            _reference.EXACT_BIT_LIMIT = saved

    def test_increasing_precision_is_what_resolves_an_ordinary_case(self):
        """A case that fails at low precision must succeed at higher."""
        saved_limit = _reference.EXACT_BIT_LIMIT
        saved_precisions = _reference.PRECISIONS
        _reference.EXACT_BIT_LIMIT = 0
        try:
            _reference.PRECISIONS = (5,)
            with self.assertRaises(UnresolvedReference):
                _reference.reference_power(1.0000001, 3.0, "float64")
            _reference.PRECISIONS = saved_precisions
            produced = _reference.reference_power(1.0000001, 3.0, "float64")
            self.assertEqual(produced.method, "enclosure")
        finally:
            _reference.EXACT_BIT_LIMIT = saved_limit
            _reference.PRECISIONS = saved_precisions


class OutsideItsDomain(unittest.TestCase):
    """The reference answers only the cases the ULP metric applies to."""

    def test_special_operands_are_refused(self):
        for base, exponent in (
            (float("nan"), 2.0),
            (2.0, float("nan")),
            (float("inf"), 2.0),
            (2.0, float("inf")),
            (2.0, float("-inf")),
            (0.0, 2.0),
            (-0.0, 3.0),
            (2.0, 0.0),
            (2.0, -0.0),
            (-2.0, 0.5),  # no real result; section 12.3.3 gives NaN
        ):
            with self.subTest(base=base, exponent=exponent):
                with self.assertRaises(OutsideReferenceDomain):
                    reference_power(base, exponent, "float64")

    def test_an_unsupported_format_is_refused(self):
        with self.assertRaises(ValueError):
            reference_power(2.0, 0.5, "float16")


class BoundaryRegions(unittest.TestCase):
    """The regions section 12.6.5 names, each resolved and recorded."""

    def resolved(self, base, exponent, dtype_name):
        produced = reference_power(base, exponent, dtype_name)
        self.assertTrue(produced.validated)
        self.assertIsInstance(produced.value, float)
        return produced

    def test_near_the_normal_boundary(self):
        for dtype_name, normal in (
            ("float32", 1.1754943508222875e-38),
            ("float64", 2.2250738585072014e-308),
        ):
            for exponent in (1.0000001, 0.9999999, 1.0):
                with self.subTest(dtype=dtype_name, exponent=exponent):
                    self.resolved(normal, exponent, dtype_name)

    def test_near_the_zero_boundary(self):
        for dtype_name, smallest in (
            ("float32", 1.401298464324817e-45),
            ("float64", 5e-324),
        ):
            for exponent in (1.0, 1.0000001, 1.5, 2.0):
                with self.subTest(dtype=dtype_name, exponent=exponent):
                    produced = self.resolved(smallest, exponent, dtype_name)
                    self.assertGreaterEqual(produced.value, 0.0)

    def test_near_overflow(self):
        for dtype_name, largest in (
            ("float32", 3.4028234663852886e38),
            ("float64", 1.7976931348623157e308),
        ):
            for exponent in (0.999, 1.0, 1.001):
                with self.subTest(dtype=dtype_name, exponent=exponent):
                    produced = self.resolved(largest, exponent, dtype_name)
                    if exponent > 1.0:
                        self.assertTrue(math.isinf(produced.value))

    def test_a_certain_overflow_short_circuit_agrees_with_the_slow_path(self):
        """The thresholds must not decide a case the long way round differs on."""
        produced = reference_power(10.0, 400.0, "float64")
        self.assertTrue(math.isinf(produced.value))
        produced = reference_power(10.0, -400.0, "float64")
        self.assertEqual(produced.value, 0.0)
        self.assertEqual(math.copysign(1.0, produced.value), 1.0)


if __name__ == "__main__":
    unittest.main()
