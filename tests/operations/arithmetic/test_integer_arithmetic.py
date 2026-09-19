"""Fixed-width integer arithmetic, against the specification.

`docs/arithmetic-semantics.md` section 4: same-dtype addition, subtraction and
multiplication preserve the dtype, wrap at the declared width, never raise on
overflow, and agree bit for bit across backends.

Every expected value comes from the wraparound rule in :mod:`_spec`, computed
from the exact mathematical result. No backend supplies an expectation.
"""

import tensors as ts

from . import _spec
from ._support import BACKENDS, ArithmeticTestCase, tensor

INTEGERS = tuple(_spec.INTEGER_DTYPES)

OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

EXACT = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def boundary_values(dtype_name):
    """Values that exercise both ends of a dtype's range."""
    low, high = _spec.integer_range(dtype_name)
    candidates = [low, low + 1, 0, 1, high - 1, high]
    if low < 0:
        candidates.append(-1)
    return sorted(set(candidates))


class WraparoundTests(ArithmeticTestCase):
    """Section 4.2: the stored result is the wrapped mathematical result."""

    def test_every_integer_dtype_wraps_at_its_boundaries(self):
        for dtype_name in INTEGERS:
            values = boundary_values(dtype_name)
            for symbol, operation in OPERATIONS.items():
                left = tensor(dtype_name, values)
                for scalar in values:
                    right = tensor(dtype_name, [scalar] * len(values))
                    expected = [
                        _spec.wrap(EXACT[symbol](a, scalar), dtype_name) for a in values
                    ]
                    with self.subTest(dtype=dtype_name, op=symbol, other=scalar):
                        self.assertIntegerResult(
                            operation(left, right), expected, dtype_name
                        )

    def test_overflow_does_not_raise(self):
        """Section 4.1: there is no overflow error for these operations."""
        for dtype_name in INTEGERS:
            low, high = _spec.integer_range(dtype_name)
            cases = (
                ("+", [high], [1]),
                ("-", [low], [1]),
                ("*", [high], [high]),
            )
            for symbol, a, b in cases:
                with self.subTest(dtype=dtype_name, op=symbol):
                    result = OPERATIONS[symbol](
                        tensor(dtype_name, a), tensor(dtype_name, b)
                    )
                    expected = [_spec.wrap(EXACT[symbol](a[0], b[0]), dtype_name)]
                    self.assertIntegerResult(result, expected, dtype_name)

    def test_the_documented_worked_examples(self):
        """The rows of section 4.3, written out."""
        rows = [
            ("uint8", "+", 255, 1, 0),
            ("uint8", "-", 0, 1, 255),
            ("uint8", "*", 16, 16, 0),
            ("int8", "+", 127, 1, -128),
            ("int8", "-", -128, 1, 127),
            ("int8", "*", -128, -1, -128),
            ("int16", "+", 32767, 1, -32768),
            ("int16", "-", -32768, 1, 32767),
            ("int16", "*", 256, 256, 0),
            ("int32", "+", 2147483647, 1, -2147483648),
            ("int32", "-", -2147483648, 1, 2147483647),
            ("int32", "*", 1000000, 1000, 1000000000),
            ("int32", "*", 65536, 65536, 0),
            ("int32", "*", 2147483647, 2147483647, 1),
            ("int32", "*", -2147483648, -1, -2147483648),
            ("int64", "+", 9223372036854775807, 1, -9223372036854775808),
            ("int64", "-", -9223372036854775808, 1, 9223372036854775807),
            ("int64", "*", 4294967296, 4294967296, 0),
        ]
        for dtype_name, symbol, a, b, expected in rows:
            with self.subTest(dtype=dtype_name, op=symbol, a=a, b=b):
                # The row and the rule must agree before the package is asked.
                self.assertEqual(_spec.wrap(EXACT[symbol](a, b), dtype_name), expected)
                result = OPERATIONS[symbol](
                    tensor(dtype_name, [a]), tensor(dtype_name, [b])
                )
                self.assertIntegerResult(result, [expected], dtype_name)

    def test_negating_the_minimum_returns_the_minimum(self):
        """A consequence of the rule, not an exception to it."""
        for dtype_name in ("int8", "int16", "int32", "int64"):
            low, _ = _spec.integer_range(dtype_name)
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [low]) * tensor(dtype_name, [-1])
                self.assertIntegerResult(result, [low], dtype_name)


class UnsignedWraparoundTests(ArithmeticTestCase):
    """Section 4.2: uint8 uses the unsigned rule, r mod 2**w."""

    def test_unsigned_arithmetic_is_modular(self):
        for symbol, operation in OPERATIONS.items():
            for a in (0, 1, 127, 128, 255):
                for b in (0, 1, 200, 255):
                    expected = _spec.wrap(EXACT[symbol](a, b), "uint8")
                    self.assertTrue(0 <= expected <= 255)
                    with self.subTest(op=symbol, a=a, b=b):
                        self.assertIntegerResult(
                            operation(tensor("uint8", [a]), tensor("uint8", [b])),
                            [expected],
                            "uint8",
                        )

    def test_unsigned_subtraction_never_produces_a_negative(self):
        result = tensor("uint8", [0, 1, 5]) - tensor("uint8", [1, 3, 10])
        self.assertIntegerResult(result, [255, 254, 251], "uint8")


class ResultDtypeTests(ArithmeticTestCase):
    """Section 4.1: same-dtype integer arithmetic preserves the dtype."""

    def test_same_dtype_operands_preserve_the_dtype(self):
        for dtype_name in INTEGERS:
            for symbol, operation in OPERATIONS.items():
                with self.subTest(dtype=dtype_name, op=symbol):
                    result = operation(tensor(dtype_name, [3]), tensor(dtype_name, [2]))
                    self.assertDtypeIs(result, dtype_name)

    def test_the_result_is_stored_at_its_declared_width(self):
        """Section 3.4: a declared dtype is the stored dtype."""
        for dtype_name in INTEGERS:
            with self.subTest(dtype=dtype_name):
                result = tensor(dtype_name, [3]) + tensor(dtype_name, [4])
                buffer = getattr(result._storage, "buffer", None)
                itemsize = getattr(buffer, "itemsize", None)
                if itemsize is not None:
                    expected = _spec.INTEGER_DTYPES[dtype_name][0] // 8
                    self.assertEqual(itemsize, expected)


class ConstructionAndCastingTests(ArithmeticTestCase):
    """Section 4.5: wraparound is arithmetic only."""

    def test_out_of_range_construction_still_raises(self):
        cases = (
            ("uint8", 256),
            ("uint8", -1),
            ("int8", 128),
            ("int8", -129),
            ("int32", 2**31),
        )
        for dtype_name, value in cases:
            with self.subTest(dtype=dtype_name, value=value):
                with self.assertRaises((OverflowError, ValueError)):
                    tensor(dtype_name, [value])

    def test_out_of_range_casting_still_raises(self):
        source = tensor("int64", [300])
        with self.assertRaises((OverflowError, ValueError)):
            source.astype(ts.uint8)

    def test_casting_is_not_wraparound(self):
        """300 would wrap to 44; casting must refuse instead."""
        self.assertEqual(_spec.wrap(300, "uint8"), 44)
        with self.assertRaises((OverflowError, ValueError)):
            tensor("int64", [300]).astype(ts.uint8)


class CrossBackendIntegerTests(ArithmeticTestCase):
    """Section 9.3: integer results are bit-identical on every backend."""

    def test_every_backend_produces_the_specified_result(self):
        for dtype_name in INTEGERS:
            values = boundary_values(dtype_name)
            for symbol in OPERATIONS:
                expected = [
                    _spec.wrap(EXACT[symbol](a, b), dtype_name)
                    for a, b in zip(values, reversed(values))
                ]
                for backend in BACKENDS:
                    with self.subTest(dtype=dtype_name, op=symbol, backend=backend):
                        with ts.use_backend(backend):
                            left = tensor(dtype_name, values)
                            right = tensor(dtype_name, list(reversed(values)))
                            result = OPERATIONS[symbol](left, right)
                            self.assertIntegerResult(result, expected, dtype_name)


if __name__ == "__main__":
    import unittest

    unittest.main()
