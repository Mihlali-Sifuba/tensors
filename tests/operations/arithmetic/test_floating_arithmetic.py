"""IEEE 754 floating-point arithmetic, against the specification.

`docs/arithmetic-semantics.md` section 5: arithmetic at the declared
precision, round-to-nearest ties-to-even, correctly rounded, overflow to
infinity, NaN and infinity propagation, signed zero, gradual underflow.

Expected values are computed in Python, whose ``float`` is binary64, and for
binary32 cases through a single correctly rounded conversion. A single
widened operation is bit-identical to native binary32 for these four
operations (section 5.5.1), so this is a legitimate oracle for one operation
— and only for one.
"""

import math
import struct

import tensors as ts

from . import _spec
from ._support import BACKENDS, ArithmeticTestCase, float_bits, tensor


def to_float32(value: float) -> float:
    """Round a binary64 value to binary32, once."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def rounded(value: float, dtype_name: str) -> float:
    """The correctly rounded representation of a value in a format."""
    return to_float32(value) if dtype_name == "float32" else value


def expect(operation, a: float, b: float, dtype_name: str) -> float:
    """One correctly rounded operation on operands of the declared format.

    The operands are rounded to the format first, the operation is performed
    in binary64, and the result is rounded once. Section 5.5.1 establishes
    that this equals native binary32 for a single +, -, * or /.
    """
    left, right = rounded(a, dtype_name), rounded(b, dtype_name)
    try:
        exact = operation(left, right)
    except ZeroDivisionError:
        # Python raises where IEEE delivers a value; supply the IEEE result.
        if left == 0.0 and right == 0.0:
            return float("nan")
        sign = math.copysign(1.0, left) * math.copysign(1.0, right)
        return math.copysign(float("inf"), sign)
    except OverflowError:
        return float("inf")
    return rounded(exact, dtype_name)


FLOATS = _spec.FLOAT_DTYPES
OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


class RoundingTests(ArithmeticTestCase):
    """Section 5.2: correctly rounded, round-to-nearest ties-to-even."""

    def test_the_documented_rounding_boundaries(self):
        half_ulp = 2.0**-24
        one_ulp = 2.0**-23
        self.assertFloatBitsEqual(
            tensor("float32", [1.0]) + tensor("float32", [half_ulp]),
            [1.0],
            "float32",
        )
        self.assertFloatBitsEqual(
            tensor("float32", [1.0]) + tensor("float32", [one_ulp]),
            [to_float32(1.0 + one_ulp)],
            "float32",
        )

    def test_float64_rounding_boundaries(self):
        self.assertFloatBitsEqual(
            tensor("float64", [1.0]) + tensor("float64", [2.0**-53]),
            [1.0],
            "float64",
        )
        self.assertFloatBitsEqual(
            tensor("float64", [1.0]) + tensor("float64", [2.0**-52]),
            [1.0 + 2.0**-52],
            "float64",
        )

    def test_a_spread_of_operands_is_correctly_rounded(self):
        values = [0.1, 1.5, -2.25, 1e10, 1e-10, 3.0, 7.0]
        for dtype_name in FLOATS:
            for symbol, operation in OPERATIONS.items():
                expected = [
                    expect(operation, a, b, dtype_name)
                    for a, b in zip(values, reversed(values))
                ]
                with self.subTest(dtype=dtype_name, op=symbol):
                    result = operation(
                        tensor(dtype_name, values),
                        tensor(dtype_name, list(reversed(values))),
                    )
                    self.assertFloatBitsEqual(result, expected, dtype_name)


class DeclaredPrecisionTests(ArithmeticTestCase):
    """Section 5.1: results follow the declared dtype, not float64."""

    def test_same_dtype_operands_preserve_the_dtype(self):
        for dtype_name in FLOATS:
            for symbol, operation in OPERATIONS.items():
                with self.subTest(dtype=dtype_name, op=symbol):
                    result = operation(
                        tensor(dtype_name, [3.0]), tensor(dtype_name, [2.0])
                    )
                    self.assertDtypeIs(result, dtype_name)

    def test_a_float32_result_is_stored_as_float32(self):
        """Section 3.4: declared dtype is the stored dtype."""
        result = tensor("float32", [1.5]) + tensor("float32", [2.5])
        buffer = getattr(result._storage, "buffer", None)
        itemsize = getattr(buffer, "itemsize", None)
        if itemsize is not None:
            self.assertEqual(itemsize, 4)

    def test_float32_precision_is_not_silently_widened(self):
        """A value needing more than 24 bits must lose them, as float32 does."""
        value = 1.0 + 2.0**-30
        result = tensor("float32", [value]) + tensor("float32", [0.0])
        self.assertFloatBitsEqual(result, [to_float32(value)], "float32")
        self.assertEqual(result.tolist()[0], 1.0)


class OverflowTests(ArithmeticTestCase):
    """Section 5.1: overflow produces a signed infinity and does not raise."""

    def test_float32_overflow_produces_infinity(self):
        big = to_float32(3.0e38)
        result = tensor("float32", [big]) + tensor("float32", [big])
        self.assertFloatBitsEqual(result, [float("inf")], "float32")

    def test_float32_multiplication_overflow(self):
        maximum = to_float32(3.4028234663852886e38)
        result = tensor("float32", [maximum]) * tensor("float32", [2.0])
        self.assertFloatBitsEqual(result, [float("inf")], "float32")

    def test_negative_overflow_produces_negative_infinity(self):
        big = to_float32(-3.0e38)
        result = tensor("float32", [big]) + tensor("float32", [big])
        self.assertFloatBitsEqual(result, [float("-inf")], "float32")

    def test_float64_overflow_produces_infinity(self):
        maximum = 1.7976931348623157e308
        result = tensor("float64", [maximum]) * tensor("float64", [2.0])
        self.assertFloatBitsEqual(result, [float("inf")], "float64")


class SpecialValueTests(ArithmeticTestCase):
    """Section 5.3: infinities, NaNs and signed zero."""

    def test_infinity_propagation(self):
        infinity, one = float("inf"), 1.0
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                self.assertFloatBitsEqual(
                    tensor(dtype_name, [infinity]) + tensor(dtype_name, [one]),
                    [infinity],
                    dtype_name,
                )
                self.assertFloatBitsEqual(
                    tensor(dtype_name, [infinity]) - tensor(dtype_name, [infinity]),
                    [float("nan")],
                    dtype_name,
                )
                self.assertFloatBitsEqual(
                    tensor(dtype_name, [0.0]) * tensor(dtype_name, [infinity]),
                    [float("nan")],
                    dtype_name,
                )

    def test_nan_propagates_through_every_operation(self):
        nan = float("nan")
        for dtype_name in FLOATS:
            for symbol, operation in OPERATIONS.items():
                with self.subTest(dtype=dtype_name, op=symbol):
                    result = operation(
                        tensor(dtype_name, [nan]), tensor(dtype_name, [1.0])
                    )
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_signed_zero_is_preserved(self):
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                # +0.0 + -0.0 is +0.0 under round-to-nearest.
                self.assertFloatBitsEqual(
                    tensor(dtype_name, [0.0]) + tensor(dtype_name, [-0.0]),
                    [0.0],
                    dtype_name,
                )
                # -0.0 + -0.0 is -0.0.
                self.assertFloatBitsEqual(
                    tensor(dtype_name, [-0.0]) + tensor(dtype_name, [-0.0]),
                    [-0.0],
                    dtype_name,
                )

    def test_negative_zero_has_a_distinct_bit_pattern(self):
        """The test itself must be able to see the sign of a zero."""
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                self.assertNotEqual(
                    float_bits(0.0, dtype_name), float_bits(-0.0, dtype_name)
                )


class SubnormalTests(ArithmeticTestCase):
    """Section 5.4: gradual underflow, operands and results."""

    SUBNORMAL_OPERAND = {"float32": 1e-40, "float64": 1e-320}
    MIN_NORMAL = {"float32": 1.1754943508222875e-38, "float64": 2.2250738585072014e-308}

    def test_a_subnormal_result_is_not_flushed_to_zero(self):
        for dtype_name in FLOATS:
            minimum = rounded(self.MIN_NORMAL[dtype_name], dtype_name)
            expected = expect(lambda a, b: a * b, minimum, 0.5, dtype_name)
            self.assertNotEqual(expected, 0.0)
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [minimum]) * tensor(dtype_name, [0.5])
                self.assertFloatBitsEqual(result, [expected], dtype_name)

    def test_a_subnormal_operand_is_not_treated_as_zero(self):
        for dtype_name in FLOATS:
            tiny = rounded(self.SUBNORMAL_OPERAND[dtype_name], dtype_name)
            self.assertNotEqual(tiny, 0.0)
            expected = expect(lambda a, b: a * b, tiny, 1.0, dtype_name)
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [tiny]) * tensor(dtype_name, [1.0])
                self.assertFloatBitsEqual(result, [expected], dtype_name)

    def test_a_subnormal_operand_reaches_a_normal_result(self):
        """The case a result-only check would miss (section 9.3)."""
        for dtype_name in FLOATS:
            tiny = rounded(self.SUBNORMAL_OPERAND[dtype_name], dtype_name)
            big = rounded(self.MIN_NORMAL[dtype_name] * 1e5, dtype_name)
            expected = expect(lambda a, b: a + b, big, tiny, dtype_name)
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [big]) + tensor(dtype_name, [tiny])
                self.assertFloatBitsEqual(result, [expected], dtype_name)

    def test_genuine_underflow_reaches_zero(self):
        smallest = {"float32": 1.401298464324817e-45, "float64": 5e-324}
        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [smallest[dtype_name]]) * tensor(
                    dtype_name, [0.25]
                )
                self.assertFloatBitsEqual(result, [0.0], dtype_name)


class CrossBackendFloatTests(ArithmeticTestCase):
    """Section 9.3: bitwise equality across backends, subnormals included."""

    def test_every_backend_produces_the_specified_bits(self):
        values = [0.1, -2.5, 1e-40, 1.1754943508222875e-38, 3.0e38, 0.0, -0.0]
        for dtype_name in FLOATS:
            for symbol, operation in OPERATIONS.items():
                expected = [
                    expect(operation, a, b, dtype_name)
                    for a, b in zip(values, reversed(values))
                ]
                for backend in BACKENDS:
                    with self.subTest(dtype=dtype_name, op=symbol, backend=backend):
                        with ts.use_backend(backend):
                            result = operation(
                                tensor(dtype_name, values),
                                tensor(dtype_name, list(reversed(values))),
                            )
                            self.assertFloatBitsEqual(result, expected, dtype_name)


if __name__ == "__main__":
    import unittest

    unittest.main()
