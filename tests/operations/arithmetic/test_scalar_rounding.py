"""Rule S3: a Python float rounds to the tensor's dtype before evaluation.

`docs/arithmetic-semantics.md` section 6.5, rule S3, and section 12.5 for
power. S3 requires a Python ``float`` to be rounded to the tensor's floating
dtype under round-to-nearest, ties-to-even *before* the operation evaluates.
The operand the backends receive must therefore be the converted value, not
the literal that was written.

Expected values come from :func:`_spec.round_to_format`, which derives the
rounding from the format's precision and exponent range using exact integer
arithmetic. No conversion performed by the package, by ``array`` or by
``struct`` is used as the source of an expectation.

Comparison is by bit pattern, so a signed zero is distinguished. NaN is
compared by classification only, its payload and sign being unspecified.
"""

import math
import struct
import unittest
from unittest.mock import patch

import tensors as ts

from . import _spec
from ._support import BACKENDS, DTYPE, ArithmeticTestCase, float_bits, tensor

#: Section 3.2 boundaries of binary32, written out rather than computed, so
#: the tests state the values they are about.
MAX_FLOAT32 = 3.4028234663852886e38
MIN_NORMAL_32 = 1.1754943508222875e-38
SMALLEST_32 = 1.401298464324817e-45

#: Halfway between the largest finite binary32 value and the first power of
#: two it cannot reach. At or above this, round-to-nearest delivers infinity
#: and S3 refuses the literal.
OVERFLOW_MIDPOINT = float(2**128 - 2**103)

#: The reported discrepancy: an exponent one binary32 ulp above one.
REPORTED_EXPONENT = 1.0000001

#: Literals that binary32 cannot hold exactly, used across the operators.
INEXACT = (0.1, 1.0000001, 3.3, 1e-40, -1e-40, 2.7182818284590451)


class S3TestCase(ArithmeticTestCase):
    """Bitwise comparison against the derived rounding."""

    def assertBitsEqual(self, produced, expected, dtype_name, context=""):
        if expected != expected:
            self.assertTrue(
                produced != produced, f"{context}: expected NaN, got {produced!r}"
            )
            return
        self.assertEqual(
            float_bits(produced, dtype_name),
            float_bits(expected, dtype_name),
            f"{context}: expected {expected!r}, got {produced!r}",
        )

    def assertScalarIs(self, produced, expected, context=""):
        """Compare two Python floats exactly.

        Deliberately *not* :meth:`assertBitsEqual` with ``float32``: that
        rounds both sides before comparing, so it cannot tell a converted
        scalar from the literal it came from, which is the whole question
        here. A converted scalar already equals its rounded value as a
        binary64 number, so the comparison is made in that format.
        """
        if expected != expected:
            self.assertTrue(
                produced != produced, f"{context}: expected NaN, got {produced!r}"
            )
            return
        self.assertEqual(
            float_bits(produced, "float64"),
            float_bits(expected, "float64"),
            f"{context}: expected {expected!r}, got {produced!r}",
        )

    def convert(self, value, dtype_name):
        from tensors.dtype import convert_scalar

        return convert_scalar(value, DTYPE[dtype_name])

    def assertConverts(self, value, dtype_name, context=""):
        """The converted scalar is what the derived rounding gives."""
        self.assertScalarIs(
            self.convert(value, dtype_name),
            _spec.round_to_format(value, dtype_name),
            context or repr(value),
        )


class TheConvertedValue(S3TestCase):
    """Section 6.5, S3 — ``convert_scalar`` returns the converted value."""

    def test_the_reported_literal_is_rounded(self):
        """The defect: the literal was returned instead of its binary32 value."""
        converted = self.convert(REPORTED_EXPONENT, "float32")
        expected = _spec.round_to_format(REPORTED_EXPONENT, "float32")

        self.assertNotEqual(
            converted,
            REPORTED_EXPONENT,
            "convert_scalar returned the original binary64 literal",
        )
        self.assertScalarIs(converted, expected)
        # Stated concretely, so the test says what the contract is.
        self.assertEqual(converted, 1.0000001192092896)

    def test_a_representable_literal_is_unchanged(self):
        for value in (0.0, 1.0, 1.5, 2.0, -0.25, 0.125, 65504.0, 2.0**-100):
            with self.subTest(value=value):
                self.assertEqual(self.convert(value, "float32"), value)
                self.assertConverts(value, "float32")

    def test_an_unrepresentable_literal_is_rounded(self):
        for value in INEXACT:
            with self.subTest(value=value):
                converted = self.convert(value, "float32")
                self.assertNotEqual(converted, value, "the literal was returned")
                self.assertConverts(value, "float32")

    def test_ties_go_to_even(self):
        """Exactly halfway between two binary32 neighbours, in both directions."""
        cases = (
            # Halfway between 1 and its successor; 1 has an even significand.
            ("down to an even significand", 1.0 + 2.0**-24, 1.0),
            # Halfway between the successor and the one after; the upper
            # neighbour is the even one here.
            ("up to an even significand", 1.0 + 3 * 2.0**-24, 1.0 + 2.0**-22),
            # Subnormal ties use the same rule at the subnormal quantum.
            ("down to zero", 2.0**-150, 0.0),
            ("up to two quanta", 3 * 2.0**-150, 2 * SMALLEST_32),
        )
        for label, value, expected in cases:
            with self.subTest(case=label):
                self.assertScalarIs(self.convert(value, "float32"), expected, label)
                self.assertConverts(value, "float32", label)

    def test_either_side_of_a_tie(self):
        """A tie is the only value that does not round to its nearer neighbour."""
        tie = 1.0 + 2.0**-24
        for label, value, expected in (
            ("below", math.nextafter(tie, 0.0), 1.0),
            ("above", math.nextafter(tie, 2.0), 1.0 + 2.0**-23),
        ):
            with self.subTest(case=label):
                self.assertScalarIs(self.convert(value, "float32"), expected, label)

    def test_the_overflow_boundary(self):
        """S3 permits a literal only while its rounded value is finite."""
        below = math.nextafter(OVERFLOW_MIDPOINT, 0.0)
        for label, value in (
            ("the largest finite value", MAX_FLOAT32),
            ("just below the midpoint", below),
        ):
            with self.subTest(case=label):
                self.assertScalarIs(self.convert(value, "float32"), MAX_FLOAT32, label)
                self.assertConverts(value, "float32", label)

        for label, value in (
            ("the midpoint itself", OVERFLOW_MIDPOINT),
            ("above the midpoint", math.nextafter(OVERFLOW_MIDPOINT, math.inf)),
            ("far above", 1e39),
            ("negative, above the midpoint", -1e39),
        ):
            with self.subTest(case=label):
                # The derived rounding agrees that these reach infinity.
                self.assertTrue(math.isinf(_spec.round_to_format(value, "float32")))
                with self.assertRaises(TypeError):
                    self.convert(value, "float32")

    def test_signed_zero_is_preserved(self):
        for dtype_name in _spec.FLOAT_DTYPES:
            for value in (0.0, -0.0):
                with self.subTest(dtype=dtype_name, value=repr(value)):
                    self.assertScalarIs(self.convert(value, dtype_name), value)

    def test_subnormal_literals(self):
        for value in (
            SMALLEST_32,
            -SMALLEST_32,
            4 * SMALLEST_32,
            1.1754942106924411e-38,  # the largest binary32 subnormal
            -1.1754942106924411e-38,
            math.nextafter(MIN_NORMAL_32, 0.0),
        ):
            with self.subTest(value=value):
                self.assertConverts(value, "float32")

    def test_literals_that_round_to_signed_zero(self):
        """Underflow keeps the sign; it does not collapse both zeros to one."""
        for value, expected in ((1e-50, 0.0), (-1e-50, -0.0), (-(2.0**-151), -0.0)):
            with self.subTest(value=value):
                converted = self.convert(value, "float32")
                self.assertScalarIs(converted, expected)
                self.assertEqual(
                    math.copysign(1.0, converted), math.copysign(1.0, value)
                )

    def test_nan_and_the_infinities_are_permitted(self):
        for dtype_name in _spec.FLOAT_DTYPES:
            with self.subTest(dtype=dtype_name):
                self.assertTrue(math.isnan(self.convert(math.nan, dtype_name)))
                self.assertEqual(self.convert(math.inf, dtype_name), math.inf)
                self.assertEqual(self.convert(-math.inf, dtype_name), -math.inf)

    def test_float64_is_unchanged(self):
        """Binary64 is the literal's own format, so conversion is identity."""
        values = INEXACT + (
            0.0,
            -0.0,
            MAX_FLOAT32,
            OVERFLOW_MIDPOINT,
            1e300,
            5e-324,
            -5e-324,
            REPORTED_EXPONENT,
        )
        for value in values:
            with self.subTest(value=value):
                converted = self.convert(value, "float64")
                self.assertEqual(
                    float_bits(converted, "float64"), float_bits(value, "float64")
                )

    def test_conversion_rounds_once(self):
        """Converting a converted value again changes nothing."""
        for value in INEXACT + (2.0**-150, OVERFLOW_MIDPOINT / 2, MAX_FLOAT32):
            with self.subTest(value=value):
                once = self.convert(value, "float32")
                twice = self.convert(once, "float32")
                self.assertScalarIs(twice, once)

    def test_a_dense_sample_matches_the_derived_rounding(self):
        """Sweeps through the regions where rounding is easiest to get wrong."""
        values = []
        value = SMALLEST_32
        for _ in range(400):  # up through the subnormals
            values.extend((value, -value))
            value = math.nextafter(value * 1.00003, math.inf)
        value = MIN_NORMAL_32
        for _ in range(400):  # either side of the normal boundary
            values.extend((value, -value))
            value = math.nextafter(value, 0.0)
        value = 1.0
        for _ in range(400):  # ordinary magnitudes, every binary64 neighbour
            values.extend((value, -value))
            value = math.nextafter(value, 2.0)
        for value in values:
            self.assertScalarIs(
                self.convert(value, "float32"),
                _spec.round_to_format(value, "float32"),
                repr(value),
            )

    def test_the_other_scalar_rules_are_untouched(self):
        """S1, S2 and S4 keep their own conversions; only S3 changed."""
        self.assertEqual(self.convert(7, "int32"), 7)  # S1
        self.assertEqual(self.convert(7.0, "int32"), 7)  # S2
        self.assertIsInstance(self.convert(7.0, "int32"), int)
        self.assertEqual(self.convert(7, "float32"), 7.0)  # S4
        with self.assertRaises(TypeError):  # S4 refuses an inexact integer
            self.convert(2**25 + 1, "float32")
        with self.assertRaises(TypeError):  # S2 refuses a non-integral float
            self.convert(7.5, "int32")


class EveryBackendEvaluatesTheConvertedOperand(S3TestCase):
    """The converted scalar is the operand, on every backend (section 8.5)."""

    #: A base with an exactly representable value, so only the scalar's
    #: conversion is under test.
    BASE = (1.5, 2.25, 0.75)

    def rounded_tensor(self, value, size):
        """The same scalar, supplied as an explicitly rounded float32 tensor."""
        return tensor("float32", [_spec.round_to_format(value, "float32")] * size)

    def test_the_reported_power_discrepancy(self):
        """`min_normal ** 1.0000001` disagreed between Python and the rest."""
        expected_exponent = _spec.round_to_format(REPORTED_EXPONENT, "float32")
        produced = {}
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = tensor("float32", [MIN_NORMAL_32])
                scalar_form = (base**REPORTED_EXPONENT).tolist()[0]
                tensor_form = (base ** tensor("float32", [expected_exponent])).tolist()[
                    0
                ]
                # Not "close": the same operands, therefore the same result.
                self.assertBitsEqual(
                    scalar_form, tensor_form, "float32", f"{backend}: scalar vs tensor"
                )
                produced[backend] = scalar_form

        first = BACKENDS[0]
        for backend in BACKENDS[1:]:
            self.assertBitsEqual(
                produced[backend], produced[first], "float32", f"{backend} vs {first}"
            )

    def operators(self):
        """Every tensor-scalar form and its reflection; S3 is not power's rule."""
        return {
            "t + s": lambda t, s: t + s,
            "s + t": lambda t, s: s + t,
            "t - s": lambda t, s: t - s,
            "s - t": lambda t, s: s - t,
            "t * s": lambda t, s: t * s,
            "s * t": lambda t, s: s * t,
            "t / s": lambda t, s: t / s,
            "s / t": lambda t, s: s / t,
            "t ** s": lambda t, s: t**s,
            "s ** t": lambda t, s: s**t,
        }

    def test_a_scalar_operand_equals_a_rounded_tensor_operand(self):
        for label, form in self.operators().items():
            for value in INEXACT:
                for backend in BACKENDS:
                    with self.subTest(operator=label, scalar=value, backend=backend):
                        with ts.use_backend(backend):
                            base = tensor("float32", self.BASE)
                            operand = self.rounded_tensor(value, len(self.BASE))
                            scalar_form = form(base, value).tolist()
                            tensor_form = form(base, operand).tolist()
                        for index, (got, want) in enumerate(
                            zip(scalar_form, tensor_form)
                        ):
                            self.assertBitsEqual(
                                got, want, "float32", f"{label} element {index}"
                            )

    def test_the_backends_agree_elementwise(self):
        for label, form in self.operators().items():
            for value in INEXACT:
                produced = {}
                for backend in BACKENDS:
                    with ts.use_backend(backend):
                        produced[backend] = form(
                            tensor("float32", self.BASE), value
                        ).tolist()
                first = BACKENDS[0]
                for backend in BACKENDS[1:]:
                    with self.subTest(operator=label, scalar=value, backend=backend):
                        for index, (got, want) in enumerate(
                            zip(produced[backend], produced[first])
                        ):
                            self.assertBitsEqual(
                                got,
                                want,
                                "float32",
                                f"{label} element {index}: {backend} vs {first}",
                            )

    def test_float64_operands_are_unaffected(self):
        """Binary64 rounds to itself, so nothing about it changes."""
        for label, form in self.operators().items():
            for value in INEXACT:
                for backend in BACKENDS:
                    with self.subTest(operator=label, scalar=value, backend=backend):
                        with ts.use_backend(backend):
                            base = tensor("float64", self.BASE)
                            scalar_form = form(base, value).tolist()
                            tensor_form = form(
                                base, tensor("float64", [value] * len(self.BASE))
                            ).tolist()
                        for index, (got, want) in enumerate(
                            zip(scalar_form, tensor_form)
                        ):
                            self.assertBitsEqual(
                                got, want, "float64", f"{label} element {index}"
                            )


class TheGraphRecordsTheConvertedScalar(S3TestCase):
    """Replay and fusion must not reconstruct the original literal."""

    #: Above the CUDA fusion threshold.
    FUSED_SIZE = 16_384

    def expressions(self):
        return {
            "v + s": lambda v, s: v + s,
            "s + v": lambda v, s: s + v,
            "v - s": lambda v, s: v - s,
            "s - v": lambda v, s: s - v,
            "v * s": lambda v, s: v * s,
            "s * v": lambda v, s: s * v,
            "v / s": lambda v, s: v / s,
            "s / v": lambda v, s: s / v,
            "v ** s": lambda v, s: v**s,
            "s ** v": lambda v, s: s**v,
        }

    def test_the_recorded_scalar_tensor_holds_the_converted_value(self):
        """A scalar enters the graph as a one-element tensor of the dtype."""
        import tensors.tensor as tensor_module

        expected = _spec.round_to_format(REPORTED_EXPONENT, "float32")
        original = tensor_module.Tensor._from_values.__func__
        recorded = []

        def recording(cls, values, dtype, shape):
            recorded.append(tuple(values))
            return original(cls, values, dtype, shape)

        with ts.use_backend("python"):
            variable = ts.Variable(tensor("float32", [1.0]), requires_grad=False)
            tensor_module.Tensor._from_values = classmethod(recording)
            try:
                for label, build in self.expressions().items():
                    recorded.clear()
                    build(variable, REPORTED_EXPONENT)
                    with self.subTest(expression=label):
                        self.assertTrue(recorded, "no scalar operand was recorded")
                        for values in recorded:
                            for value in values:
                                self.assertScalarIs(value, expected, label)
            finally:
                tensor_module.Tensor._from_values = classmethod(original)

    def test_replay_matches_eager(self):
        from tensors.graph import Computation

        for label, build in self.expressions().items():
            for value in INEXACT:
                for backend in BACKENDS:
                    with self.subTest(expression=label, scalar=value, backend=backend):
                        with ts.use_backend(backend):
                            base = tensor("float32", [1.5, 2.25, 0.75])
                            eager = build(base, value).tolist()
                            variable = ts.Variable(base, requires_grad=False)
                            replay = (
                                Computation(build(variable, value)).forward().tolist()
                            )
                        for index, (got, want) in enumerate(zip(replay, eager)):
                            self.assertBitsEqual(
                                got, want, "float32", f"{label} element {index}"
                            )

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_the_fused_kernel_receives_the_converted_operand(self):
        """The fused plan's scalar operand carries the rounded value.

        A scalar always reaches a fused plan as a one-element tensor operand:
        the planner leaves a step's literal scalar field unset, so there is no
        second copy of the literal for a kernel to rebuild from. What the
        kernel receives is asserted directly rather than assumed.
        """
        import tensors.backend as backend_module
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading
        from tensors.graph import Computation

        original = backend_module.execute_fused_elementwise

        for label, build in self.expressions().items():
            for value in INEXACT:
                expected = _spec.round_to_format(value, "float32")
                received = []

                def recording(sources, steps, **keywords):
                    received.append((sources, steps))
                    return original(sources, steps, **keywords)

                with self.subTest(expression=label, scalar=value):
                    with ts.use_backend("cuda"):
                        base = tensor("float32", [1.5] * self.FUSED_SIZE)
                        eager = (build(base, value) * 1.0).tolist()[0]
                        variable = ts.Variable(base, requires_grad=False)
                        with (
                            patch.object(
                                backend_module, "execute_fused_elementwise", recording
                            ),
                            patch.object(
                                cuda_backend,
                                "fused_elementwise",
                                wraps=cuda_backend.fused_elementwise,
                            ) as fused,
                        ):
                            loading._clear_backend_kernel_cache()
                            produced = (
                                Computation(build(variable, value) * 1.0)
                                .forward()
                                .tolist()[0]
                            )
                        self.assertTrue(
                            fused.called, "the fused CUDA kernel was not reached"
                        )
                    self.assertBitsEqual(produced, eager, "float32", f"{label} fused")

                    self.assertTrue(received, "the fused plan was not built")
                    sources, steps = received[0]
                    for _, scalar, _, _ in steps:
                        self.assertIsNone(
                            scalar, "a fused step carried a literal scalar"
                        )
                    scalars = [
                        source.tolist()[0] for source in sources if source.size == 1
                    ]
                    self.assertTrue(scalars, "no scalar operand reached the kernel")
                    self.assertIn(
                        float_bits(expected, "float32"),
                        [float_bits(value, "float32") for value in scalars],
                        f"{label}: the fused operand is not the converted scalar",
                    )


@unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
class ConversionInspectsNoTensor(S3TestCase):
    """Section 6.4 — conversion reads the scalar, never the tensor."""

    def test_no_element_is_read_from_the_device(self):
        import tensors.tensor as tensor_module

        reads = 0
        original = tensor_module.Tensor._data.fget

        def counted(self):
            nonlocal reads
            reads += 1
            return original(self)

        with ts.use_backend("cuda"):
            base = ts.full((4096,), 1.5, dtype=ts.float32) + 0.0
            tensor_module.Tensor._data = property(counted)
            try:
                for value in INEXACT:
                    for form in (
                        lambda t, s: t + s,
                        lambda t, s: t * s,
                        lambda t, s: t**s,
                        lambda t, s: s / t,
                        lambda t, s: s**t,
                    ):
                        result = form(base, value)
                        self.assertEqual(
                            type(result.backend_storage).__name__, "CudaStorage"
                        )
            finally:
                tensor_module.Tensor._data = property(original)
        self.assertEqual(reads, 0, "scalar conversion materialised a tensor")


if __name__ == "__main__":
    unittest.main()
