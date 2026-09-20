"""Fixed-width integer exponentiation (D3, specification §12.4.1).

Expected values come from exact arithmetic on Python integers:
``wrap(x ** n)`` for ordinary exponents, where ``x ** n`` is computed at full
precision and only then reduced to the declared width. NumPy's and CuPy's own
power implementations are never consulted.

Only the very large exponents use ``pow(x, n, m)``, because ``3 ** 1000000``
has more than a million bits; that is still exact modular arithmetic, and the
test that matters for it is that the package does not build the big integer
either.
"""

import unittest

import tensors as ts

from . import _spec
from ._support import BACKENDS, DTYPE, ArithmeticTestCase, tensor

INTEGERS = tuple(_spec.INTEGER_DTYPES)


def exact(base: int, exponent: int, dtype_name: str) -> int:
    """``x ** n`` at full precision, then reduced to the declared width."""
    return _spec.wrap(base**exponent, dtype_name)


def exact_modular(base: int, exponent: int, dtype_name: str) -> int:
    """The same value, for exponents too large to expand."""
    low, high = _spec.integer_range(dtype_name)
    return _spec.wrap(pow(base, exponent, high - low + 1), dtype_name)


def representable(dtype_name: str, values):
    low, high = _spec.integer_range(dtype_name)
    return [value for value in values if low <= value <= high]


class WraparoundAtEachWidth(ArithmeticTestCase):
    """§12.4.1 — the result wraps rather than raising or promoting."""

    #: The four worked examples the specification states.
    SPECIFIED = (
        ("int32", 2, 31, -2147483648),
        ("uint8", 16, 2, 0),
        ("int8", -2, 7, -128),
        ("int32", 0, 0, 1),
    )

    def test_the_specified_examples(self):
        for name, base, exponent, expected in self.SPECIFIED:
            for backend in BACKENDS:
                with self.subTest(
                    dtype=name, base=base, exponent=exponent, backend=backend
                ):
                    with ts.use_backend(backend):
                        result = tensor(name, [base]) ** exponent
                    self.assertIs(result.dtype, DTYPE[name])
                    self.assertEqual(int(result.tolist()[0]), expected)
                    self.assertEqual(expected, exact(base, exponent, name))

    def test_overflow_wraps_at_every_width(self):
        """The first exponent that leaves each dtype's range."""
        for name in INTEGERS:
            low, high = _spec.integer_range(name)
            base = 2
            exponent = (high).bit_length()  # 2 ** bit_length exceeds `high`
            self.assertGreater(2**exponent, high)
            for backend in BACKENDS:
                with self.subTest(dtype=name, backend=backend):
                    with ts.use_backend(backend):
                        result = tensor(name, [base]) ** exponent
                    self.assertIs(result.dtype, DTYPE[name])
                    self.assertEqual(
                        int(result.tolist()[0]), exact(base, exponent, name)
                    )

    def test_overflow_never_raises_or_promotes(self):
        for name in INTEGERS:
            for backend in BACKENDS:
                with self.subTest(dtype=name, backend=backend):
                    with ts.use_backend(backend):
                        result = tensor(name, [2, 3]) ** 64
                    self.assertIs(result.dtype, DTYPE[name])
                    self.assertEqual(
                        [int(v) for v in result.tolist()],
                        [exact(2, 64, name), exact(3, 64, name)],
                    )

    def test_wraparound_is_not_saturation(self):
        """`uint8: 16 ** 2` is 0, not 255."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                self.assertEqual((tensor("uint8", [16]) ** 2).tolist(), [0])
                self.assertEqual((tensor("uint8", [255]) ** 2).tolist(), [1])


class ExhaustiveAgreement(ArithmeticTestCase):
    """Every dtype, a spread of bases and exponents, against exact arithmetic."""

    BASES = (0, 1, -1, 2, -2, 3, -3, 7, 10, 16, 100, 127, 128, 255, -128)
    EXPONENTS = (0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 100)

    def test_every_dtype_and_exponent(self):
        for name in INTEGERS:
            bases = representable(name, self.BASES)
            for exponent in self.EXPONENTS:
                expected = [exact(b, exponent, name) for b in bases]
                for backend in BACKENDS:
                    with self.subTest(dtype=name, exponent=exponent, backend=backend):
                        with ts.use_backend(backend):
                            result = tensor(name, bases) ** exponent
                        self.assertIs(result.dtype, DTYPE[name])
                        self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_negative_bases_with_non_negative_exponents(self):
        """Sign follows the parity of the exponent, then wraps."""
        for name in ("int8", "int16", "int32", "int64"):
            bases = representable(name, (-1, -2, -3, -7))
            for exponent in (0, 1, 2, 3, 4, 5, 8, 9, 31, 32):
                expected = [exact(b, exponent, name) for b in bases]
                for backend in BACKENDS:
                    with self.subTest(dtype=name, exponent=exponent, backend=backend):
                        with ts.use_backend(backend):
                            result = tensor(name, bases) ** exponent
                        self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_zero_and_unit_exponents(self):
        for name in INTEGERS:
            bases = representable(name, (0, 1, 2, 7, 100))
            for exponent, description in (
                (0, "anything to the zeroth is one"),
                (1, "anything to the first is itself"),
            ):
                for backend in BACKENDS:
                    with self.subTest(
                        dtype=name, exponent=exponent, backend=backend, note=description
                    ):
                        with ts.use_backend(backend):
                            result = tensor(name, bases) ** exponent
                        self.assertEqual(
                            [int(v) for v in result.tolist()],
                            [exact(b, exponent, name) for b in bases],
                        )

    def test_zero_to_the_zeroth_is_one(self):
        for name in INTEGERS:
            for backend in BACKENDS:
                with self.subTest(dtype=name, backend=backend):
                    with ts.use_backend(backend):
                        self.assertEqual((tensor(name, [0]) ** 0).tolist(), [1])

    def test_all_backends_produce_identical_integers(self):
        if len(BACKENDS) < 2:
            self.skipTest("needs more than one backend")
        for name in INTEGERS:
            bases = representable(name, self.BASES)
            for exponent in (0, 3, 17, 64):
                results = {}
                for backend in BACKENDS:
                    with ts.use_backend(backend):
                        results[backend] = [
                            int(v) for v in (tensor(name, bases) ** exponent).tolist()
                        ]
                reference = BACKENDS[0]
                for backend in BACKENDS[1:]:
                    with self.subTest(dtype=name, exponent=exponent, backend=backend):
                        self.assertEqual(results[backend], results[reference])


class OperandForms(ArithmeticTestCase):
    """Tensor, scalar, reflected, broadcast, view and Variable operands."""

    def test_tensor_tensor(self):
        bases, exponents = [2, 3, 4, 5], [10, 20, 15, 13]
        expected = [exact(b, e, "int32") for b, e in zip(bases, exponents)]
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                result = tensor("int32", bases) ** tensor("int32", exponents)
                self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_reflected_scalar_base(self):
        exponents = [31, 32, 33]
        expected = [exact(2, e, "int32") for e in exponents]
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                result = 2 ** tensor("int32", exponents)
                self.assertIs(result.dtype, ts.int32)
                self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_broadcasting(self):
        expected = [
            exact(b, e, "int32") for b, e in zip([2, 3, 4, 5], [10, 10, 20, 20])
        ]
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                grid = ts.Tensor([[2, 3], [4, 5]], dtype=ts.int32)
                column = ts.Tensor([[10], [20]], dtype=ts.int32)
                result = grid**column
                self.assertEqual(result.shape, (2, 2))
                self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_a_view_raises_only_its_own_elements(self):
        expected = [exact(4, 20, "int32"), exact(5, 20, "int32")]
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                grid = ts.Tensor([[2, 3], [4, 5]], dtype=ts.int32)
                result = grid[1] ** tensor("int32", [20, 20])
                self.assertEqual([int(v) for v in result.tolist()], expected)

    def test_integer_variables(self):
        bases, exponents = [2, 3], [40, 40]
        expected = [exact(b, e, "int32") for b, e in zip(bases, exponents)]
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(tensor("int32", bases), requires_grad=False)
                exponent = ts.Variable(tensor("int32", exponents), requires_grad=False)
                self.assertEqual(
                    [int(v) for v in (base**exponent).data.tolist()], expected
                )
                self.assertEqual([int(v) for v in (base**40).data.tolist()], expected)


class PromotionHappensBeforeWraparound(ArithmeticTestCase):
    """§12.4.1 — the width is the promoted result width, not the base's."""

    def test_mixed_integer_dtypes_wrap_at_the_promoted_width(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                # uint8 with int8 promotes to int16, where 16 ** 2 = 256 fits.
                promoted = tensor("uint8", [16]) ** tensor("int8", [2])
                self.assertIs(promoted.dtype, ts.int16)
                self.assertEqual([int(v) for v in promoted.tolist()], [256])

                # The same base in its own width wraps to zero.
                narrow = tensor("uint8", [16]) ** 2
                self.assertIs(narrow.dtype, ts.uint8)
                self.assertEqual([int(v) for v in narrow.tolist()], [0])

    def test_every_mixed_pair_wraps_at_its_promoted_width(self):
        for base_name in INTEGERS:
            for exponent_name in INTEGERS:
                promoted = _spec.promote(base_name, exponent_name)
                if promoted == _spec.CAST:
                    continue
                low, high = _spec.integer_range(base_name)
                base_value = min(7, high)
                with self.subTest(base=base_name, exponent=exponent_name):
                    with ts.use_backend(BACKENDS[0]):
                        result = tensor(base_name, [base_value]) ** tensor(
                            exponent_name, [9]
                        )
                    self.assertIs(result.dtype, DTYPE[promoted])
                    self.assertEqual(
                        int(result.tolist()[0]), exact(base_value, 9, promoted)
                    )


class LargeExponents(ArithmeticTestCase):
    """The algorithm is logarithmic and builds no enormous intermediate."""

    def test_very_large_exponents_are_exact(self):
        cases = ((3, 1000), (5, 4001), (7, 65537))
        for name in ("int32", "int64"):
            for base, exponent in cases:
                expected = exact_modular(base, exponent, name)
                for backend in BACKENDS:
                    with self.subTest(
                        dtype=name, base=base, exponent=exponent, backend=backend
                    ):
                        with ts.use_backend(backend):
                            result = tensor(name, [base]) ** exponent
                        self.assertEqual(int(result.tolist()[0]), expected)

    def test_an_enormous_exponent_completes_promptly(self):
        """`3 ** 1000000` would be a 1.5-million-bit integer if expanded."""
        import time

        expected = exact_modular(3, 1_000_000, "int64")
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                started = time.perf_counter()
                result = tensor("int64", [3]) ** 1_000_000
                elapsed = time.perf_counter() - started
                self.assertEqual(int(result.tolist()[0]), expected)
                self.assertLess(elapsed, 2.0)

    def test_the_python_reference_never_expands_the_power(self):
        """Exponent 2**62 is impossible to expand; modular arithmetic is not."""
        from tensors.backend.python.kernels.arithmetic.power import _integer_power

        exponent = 2**62
        self.assertEqual(
            _integer_power(3, exponent, ts.int64),
            exact_modular(3, exponent, "int64"),
        )


@unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
class CudaExecutesIntegerPowerNatively(ArithmeticTestCase):
    """The capability gap this milestone closes."""

    def _reference_module(self):
        import sys

        import tensors.backend.python.kernels.arithmetic.power  # noqa: F401

        return sys.modules["tensors.backend.python.kernels.arithmetic.power"]

    def test_the_provider_kernel_no_longer_declines(self):
        from tensors.backend.loading import _backend_kernel
        from tensors.backend.cuda.conversion import _arithmetic_operand

        with ts.use_backend("cuda"):
            base = tensor("int32", [2, 3, 4, 5])
            storage = _backend_kernel("power")(
                _arithmetic_operand(base, ts.int32),
                _arithmetic_operand(5, ts.int32),
                dtype=ts.int32,
                output_shape=(4,),
            )
        self.assertIsNotNone(storage, "the CUDA kernel declined integer operands")
        self.assertEqual(type(storage).__name__, "CudaStorage")

    def test_results_stay_device_resident_at_every_size(self):
        from unittest.mock import patch

        reference = self._reference_module()
        for size in (1, 4, 31, 32, 4096):
            with self.subTest(size=size):
                with ts.use_backend("cuda"):
                    base = ts.full((size,), 3, dtype=ts.int32) + 0
                    with patch.object(
                        reference, "power", wraps=reference.power
                    ) as fallback:
                        result = base**5
                self.assertEqual(type(result._storage).__name__, "CudaStorage")
                self.assertFalse(
                    fallback.called,
                    "integer power fell back to the Python reference",
                )
                self.assertEqual(int(result.tolist()[0]), exact(3, 5, "int32"))

    def test_every_integer_dtype_runs_on_the_device(self):
        for name in INTEGERS:
            with self.subTest(dtype=name), ts.use_backend("cuda"):
                base = tensor(name, [2, 3]) + 0
                result = base**5
                self.assertEqual(type(result._storage).__name__, "CudaStorage")
                self.assertEqual(
                    [int(v) for v in result.tolist()],
                    [exact(2, 5, name), exact(3, 5, name)],
                )


class NegativeExponentsStillRaise(ArithmeticTestCase):
    """D4 is unaffected by D3 (§12.4.2)."""

    def test_negative_exponents_raise_for_every_base(self):
        for base in (1, -1, 2, 0):
            for backend in BACKENDS:
                with self.subTest(base=base, backend=backend):
                    with ts.use_backend(backend):
                        with self.assertRaises(ValueError):
                            _ = tensor("int32", [base]) ** -1

    def test_a_negative_exponent_in_a_tensor_still_raises(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                with self.assertRaises(ValueError):
                    _ = tensor("int32", [2, 2]) ** tensor("int32", [31, -1])


if __name__ == "__main__":
    unittest.main()
