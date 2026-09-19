"""Dtype promotion and Python scalar conversion, against the specification.

`docs/arithmetic-semantics.md` section 6. Promotion depends on operand dtypes
only; a Python scalar is weakly typed and converts under rules S1 to S4.

Expected results come from :mod:`_spec`, which derives them from the stated
principles rather than from the package or from the published table.
"""

import tensors as ts

from . import _spec
from ._support import ArithmeticTestCase, tensor

BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def representative(dtype_name):
    """A small value every dtype can hold, so promotion is what is tested."""
    return tensor(dtype_name, [2])


class PromotionTableTests(ArithmeticTestCase):
    """Section 6.2: all forty-nine cells."""

    def test_every_cell_of_the_promotion_table(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                expected = _spec.promote(left, right)
                for symbol, operation in BINARY.items():
                    with self.subTest(left=left, right=right, op=symbol):
                        call = lambda: operation(
                            representative(left), representative(right)
                        )
                        if expected == _spec.CAST:
                            self.assertRaisesConversion(call)
                        else:
                            self.assertDtypeIs(call(), expected)

    def test_promotion_is_symmetric(self):
        for left in _spec.DTYPES:
            for right in _spec.DTYPES:
                if _spec.promote(left, right) == _spec.CAST:
                    continue
                with self.subTest(left=left, right=right):
                    forward = representative(left) + representative(right)
                    reflected = representative(right) + representative(left)
                    self.assertIs(forward.dtype, reflected.dtype)

    def test_the_documented_non_obvious_cells(self):
        cases = (
            ("uint8", "int8", "int16"),
            ("int32", "float32", "float64"),
            ("int16", "float32", "float32"),
            ("float32", "float64", "float64"),
            ("uint8", "int64", "int64"),
        )
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertEqual(_spec.promote(left, right), expected)
                self.assertDtypeIs(
                    representative(left) + representative(right), expected
                )

    def test_int64_with_a_floating_dtype_requires_an_explicit_cast(self):
        for floating in ("float32", "float64"):
            for symbol, operation in BINARY.items():
                with self.subTest(floating=floating, op=symbol):
                    self.assertRaisesConversion(
                        lambda: operation(
                            representative("int64"), representative(floating)
                        )
                    )
                    self.assertRaisesConversion(
                        lambda: operation(
                            representative(floating), representative("int64")
                        )
                    )

    def test_an_explicit_cast_makes_the_refused_combination_work(self):
        result = representative("int64").astype(ts.float64) + representative("float64")
        self.assertDtypeIs(result, "float64")


class ValueIndependenceTests(ArithmeticTestCase):
    """Section 6.4: promotion never inspects element values."""

    def test_small_values_in_a_wide_dtype_still_promote_by_dtype(self):
        small = tensor("int64", [1, 2, 3])
        for floating in ("float32", "float64"):
            with self.subTest(floating=floating):
                self.assertRaisesConversion(
                    lambda: small + tensor(floating, [1.0, 2.0, 3.0])
                )

    def test_the_result_dtype_does_not_depend_on_the_values(self):
        for values in ([0], [2**40], [-(2**40)]):
            with self.subTest(values=values):
                result = tensor("int32", [1]) * tensor("int16", [1])
                self.assertDtypeIs(result, _spec.promote("int32", "int16"))


class ScalarConversionTests(ArithmeticTestCase):
    """Section 6.5: rules S1 to S4."""

    #: Literals chosen to exercise each rule's boundary.
    LITERALS = (
        0,
        1,
        2,
        -1,
        255,
        256,
        3.0,
        3.5,
        0.1,
        2.0,
        -0.0,
        1e-60,
        1e300,
        float("inf"),
        float("nan"),
        2**24,
        2**24 + 1,
        2**25,
        2**25 + 1,
        2**53,
        2**53 + 1,
        2**54,
    )

    def test_every_literal_against_every_dtype(self):
        for dtype_name in _spec.DTYPES:
            for literal in self.LITERALS:
                rule, converts = _spec.scalar_rule(dtype_name, literal)
                with self.subTest(dtype=dtype_name, literal=literal, rule=rule):
                    call = lambda: tensor(dtype_name, [1]) + literal
                    if converts:
                        self.assertDtypeIs(call(), dtype_name)
                    else:
                        self.assertRaisesConversion(call)

    def test_the_documented_scalar_rows(self):
        rows = (
            ("float32", 2.0, "float32"),
            ("float32", 0.1, "float32"),
            ("float32", 1e-60, "float32"),
            ("float32", float("inf"), "float32"),
            ("float32", 1e300, "raises"),
            ("float32", 2, "float32"),
            ("float32", 2**24, "float32"),
            ("float32", 2**24 + 1, "raises"),
            ("float32", 2**25, "float32"),
            ("float32", 2**25 + 1, "raises"),
            ("float64", 2, "float64"),
            ("float64", 2**53 + 1, "raises"),
            ("float64", 2**54, "float64"),
            ("int32", 2, "int32"),
            ("int64", 1, "int64"),
            ("uint8", 255, "uint8"),
            ("uint8", 256, "raises"),
            ("uint8", -1, "raises"),
            ("int32", 3.0, "int32"),
            ("int32", 3.5, "raises"),
        )
        for dtype_name, literal, expected in rows:
            with self.subTest(dtype=dtype_name, literal=literal):
                call = lambda: tensor(dtype_name, [1]) + literal
                if expected == "raises":
                    self.assertRaisesConversion(call)
                else:
                    self.assertDtypeIs(call(), expected)

    def test_s4_uses_per_value_representability_not_the_interval(self):
        """2**25 exceeds 2**24 yet is exactly representable in float32."""
        self.assertTrue(_spec.exactly_representable(2**25, "float32"))
        self.assertFalse(_spec.exactly_representable(2**24 + 1, "float32"))
        self.assertDtypeIs(tensor("float32", [1.0]) + 2**25, "float32")
        self.assertRaisesConversion(lambda: tensor("float32", [1.0]) + (2**24 + 1))

    def test_a_scalar_never_widens_the_result(self):
        """A refused scalar raises rather than promoting the tensor."""
        self.assertRaisesConversion(lambda: tensor("int32", [1]) + 3.5)
        self.assertRaisesConversion(lambda: tensor("uint8", [1]) + 256)

    def test_booleans_are_refused(self):
        for dtype_name in _spec.DTYPES:
            with self.subTest(dtype=dtype_name):
                self.assertRaisesConversion(lambda: tensor(dtype_name, [1]) + True)

    def test_reflected_operators_follow_the_same_rules(self):
        cases = (
            ("float32", 2.0, "float32"),
            ("int32", 2, "int32"),
            ("int32", 3.5, "raises"),
            ("uint8", 256, "raises"),
            ("float32", 2**25, "float32"),
        )
        for dtype_name, literal, expected in cases:
            for symbol in ("+", "-", "*"):
                with self.subTest(dtype=dtype_name, literal=literal, op=symbol):
                    operand = tensor(dtype_name, [1])
                    call = {
                        "+": lambda: literal + operand,
                        "-": lambda: literal - operand,
                        "*": lambda: literal * operand,
                    }[symbol]
                    if expected == "raises":
                        self.assertRaisesConversion(call)
                    else:
                        self.assertDtypeIs(call(), expected)

    def test_the_scalar_value_is_used_not_rounded_into_range(self):
        """S1 converts exactly; it does not wrap an out-of-range literal."""
        self.assertEqual(_spec.wrap(256, "uint8"), 0)
        self.assertRaisesConversion(lambda: tensor("uint8", [1]) + 256)


class ScalarArithmeticValueTests(ArithmeticTestCase):
    """A converted scalar then takes part in ordinary arithmetic."""

    def test_an_integer_scalar_wraps_with_the_tensor_dtype(self):
        result = tensor("uint8", [255]) + 1
        self.assertIntegerResult(result, [0], "uint8")

    def test_a_float_scalar_rounds_to_the_tensor_dtype(self):
        result = tensor("float32", [0.0]) + 0.1
        self.assertFloatBitsEqual(result, [0.1], "float32")


if __name__ == "__main__":
    import unittest

    unittest.main()
