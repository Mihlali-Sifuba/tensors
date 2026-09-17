"""True division, against the specification.

`docs/arithmetic-semantics.md` section 7: floating division adopts IEEE
results and never raises; integer true division produces a floating dtype and
refuses the combinations that would lose integer precision; integer division
by zero raises.
"""

import math

import tensors as ts

from . import _spec
from ._support import BACKENDS, ArithmeticTestCase, tensor

FLOATS = _spec.FLOAT_DTYPES
INTEGERS = tuple(_spec.INTEGER_DTYPES)


class FloatingDivisionByZeroTests(ArithmeticTestCase):
    """Section 7.2: the IEEE results, and no ZeroDivisionError."""

    #: The table of section 7.2, written out.
    ROWS = (
        (1.0, 0.0, float("inf")),
        (1.0, -0.0, float("-inf")),
        (-1.0, 0.0, float("-inf")),
        (-1.0, -0.0, float("inf")),
        (0.0, 0.0, float("nan")),
        (float("inf"), float("inf"), float("nan")),
        (0.0, float("inf"), 0.0),
        (float("inf"), 0.0, float("inf")),
    )

    def test_the_documented_division_by_zero_rows(self):
        for dtype_name in FLOATS:
            for numerator, denominator, expected in self.ROWS:
                with self.subTest(
                    dtype=dtype_name, numerator=numerator, denominator=denominator
                ):
                    result = tensor(dtype_name, [numerator]) / tensor(
                        dtype_name, [denominator]
                    )
                    self.assertFloatBitsEqual(result, [expected], dtype_name)

    def test_division_by_zero_does_not_raise(self):
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [1.0]) / tensor(dtype_name, [0.0])
                self.assertTrue(math.isinf(result.tolist()[0]))

    def test_a_zero_scalar_denominator_does_not_raise(self):
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [1.0]) / 0.0
                self.assertTrue(math.isinf(result.tolist()[0]))

    def test_a_mixed_vector_divides_elementwise_without_raising(self):
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [1.0, 2.0, 0.0]) / tensor(
                    dtype_name, [0.0, 1.0, 0.0]
                )
                values = result.tolist()
                self.assertTrue(math.isinf(values[0]))
                self.assertEqual(values[1], 2.0)
                self.assertTrue(math.isnan(values[2]))

    def test_every_backend_agrees_on_division_by_zero(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        result = tensor(dtype_name, [1.0, -1.0, 0.0]) / tensor(
                            dtype_name, [0.0, 0.0, 0.0]
                        )
                        values = result.tolist()
                        self.assertEqual(values[0], float("inf"))
                        self.assertEqual(values[1], float("-inf"))
                        self.assertTrue(math.isnan(values[2]))


class FloatingDivisionResultTests(ArithmeticTestCase):
    """Section 7.1: same-dtype floating division preserves the dtype."""

    def test_same_dtype_division_preserves_the_dtype(self):
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [7.0]) / tensor(dtype_name, [2.0])
                self.assertDtypeIs(result, dtype_name)
                self.assertEqual(result.tolist(), [3.5])

    def test_mixed_floating_division_promotes(self):
        result = tensor("float32", [7.0]) / tensor("float64", [2.0])
        self.assertDtypeIs(result, "float64")


class IntegerTrueDivisionTests(ArithmeticTestCase):
    """Section 7.3: rule D, per dtype pair."""

    def test_every_integer_pair(self):
        for left in INTEGERS:
            for right in INTEGERS:
                expected = _spec.divide_result(left, right)
                with self.subTest(left=left, right=right, expected=expected):
                    call = lambda: tensor(left, [7]) / tensor(right, [2])
                    if expected == _spec.CAST:
                        self.assertRaisesConversion(call)
                    else:
                        result = call()
                        self.assertDtypeIs(result, expected)
                        self.assertEqual(result.tolist(), [3.5])

    def test_the_documented_rows(self):
        rows = (
            ("uint8", "uint8", "float32"),
            ("int8", "int8", "float32"),
            ("int16", "int16", "float32"),
            ("uint8", "int8", "float32"),
            ("uint8", "int16", "float32"),
            ("int8", "int16", "float32"),
            ("int32", "int32", "float64"),
            ("int16", "int32", "float64"),
            ("uint8", "int32", "float64"),
        )
        for left, right, expected in rows:
            with self.subTest(left=left, right=right):
                self.assertEqual(_spec.divide_result(left, right), expected)
                self.assertDtypeIs(tensor(left, [7]) / tensor(right, [2]), expected)

    def test_int64_division_requires_an_explicit_cast(self):
        for other in INTEGERS:
            with self.subTest(other=other):
                self.assertRaisesConversion(
                    lambda: tensor("int64", [7]) / tensor(other, [2])
                )
                self.assertRaisesConversion(
                    lambda: tensor(other, [7]) / tensor("int64", [2])
                )

    def test_an_explicit_cast_makes_int64_division_work(self):
        result = tensor("int64", [7]).astype(ts.float64) / tensor("int64", [2]).astype(
            ts.float64
        )
        self.assertDtypeIs(result, "float64")
        self.assertEqual(result.tolist(), [3.5])

    def test_division_never_produces_an_integer_dtype(self):
        for left in INTEGERS:
            for right in INTEGERS:
                if _spec.divide_result(left, right) == _spec.CAST:
                    continue
                with self.subTest(left=left, right=right):
                    result = tensor(left, [7]) / tensor(right, [2])
                    self.assertIn(result.dtype.kind, ("floating",))


class IntegerDivisionByZeroTests(ArithmeticTestCase):
    """Section 7.2: integer division by zero raises."""

    def test_integer_division_by_zero_raises(self):
        for dtype_name in INTEGERS:
            if _spec.divide_result(dtype_name, dtype_name) == _spec.CAST:
                continue
            with self.subTest(dtype=dtype_name):
                with self.assertRaises(ZeroDivisionError):
                    tensor(dtype_name, [1]) / tensor(dtype_name, [0])

    def test_an_integer_zero_scalar_denominator_raises(self):
        for dtype_name in INTEGERS:
            if _spec.divide_result(dtype_name, dtype_name) == _spec.CAST:
                continue
            with self.subTest(dtype=dtype_name):
                with self.assertRaises(ZeroDivisionError):
                    tensor(dtype_name, [1]) / 0

    def test_a_zero_anywhere_in_an_integer_denominator_raises(self):
        with self.assertRaises(ZeroDivisionError):
            tensor("int32", [1, 2, 3]) / tensor("int32", [1, 0, 3])


if __name__ == "__main__":
    import unittest

    unittest.main()
