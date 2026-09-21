"""Forward square root, against docs/sqrt-semantics.md.

Every expectation is derived from that document. Ordinary values are literals
from its tables; the correctly rounded results of section 1.2 come from
`_sqrt_reference`, which computes them from exact integer arithmetic. No
Python, NumPy or CUDA square root ever supplies an expected value.

Differentiation is outside this specification and is not covered here.
"""

import importlib
import math
import random
import struct
import unittest
from unittest.mock import patch

import tensors as ts

from tests.operations.elementary._sqrt_reference import reference_sqrt

#: The smallest positive binary32 subnormal, and its correctly rounded root.
SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45

#: The largest binary32 subnormal.
LARGEST_FLOAT32_SUBNORMAL = 1.1754942106924411e-38

#: The smallest positive binary64 subnormal.
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

FLOAT_DTYPES = (ts.float64, ts.float32)
INTEGER_DTYPES = (ts.int64, ts.int32, ts.int16, ts.int8, ts.uint8)
ALL_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES

PYTHON_SQRT_MODULE = "tensors.backend.python.kernels.elementwise.sqrt"


def available_backends():
    return ts.available_backends()


def signed_zero(value):
    """A comparable form that separates -0.0 from +0.0."""
    return (value, math.copysign(1.0, value)) if value == 0 else value


def float32_sample(count, seed):
    """Finite positive float32 values, drawn from the whole exponent range."""
    generator = random.Random(seed)
    values = []
    while len(values) < count:
        bits = generator.getrandbits(31)
        candidate = struct.unpack("<f", struct.pack("<I", bits))[0]
        if math.isfinite(candidate) and candidate > 0.0:
            values.append(candidate)
    return values


class SqrtSpecificationTests(unittest.TestCase):
    """Sections 1.1 and 1.3 to 1.7, on every backend."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_ordinary_values_on_every_float_dtype(self):
        """Section 1.2 on operands whose roots are exact."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(
                            ts.Tensor([0.0, 1.0, 4.0, 9.0, 0.25], dtype=dtype)
                        )
                    self.assertEqual(result.tolist(), [0.0, 1.0, 2.0, 3.0, 0.5])

    def test_shape_is_preserved_including_multidimensional_and_scalar(self):
        """Section 1.1: the shape is untouched."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    flat = ts.Tensor([4.0], dtype=ts.float64)
                    nested = ts.Tensor([[4.0, 9.0], [16.0, 25.0]], dtype=ts.float64)
                    self.assertEqual(ts.sqrt(flat).shape, flat.shape)
                    produced = ts.sqrt(nested)
                    self.assertEqual(produced.shape, nested.shape)
                    self.assertEqual(produced.tolist(), [2.0, 3.0, 4.0, 5.0])

    def test_a_floating_dtype_is_preserved(self):
        """Section 1.1: float32 stays float32 and float64 stays float64."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([4.0], dtype=dtype))
                    self.assertIs(result.dtype, dtype)

    def test_every_integer_dtype_converts_to_float64(self):
        """Section 1.1: the integer rule, for all five integer dtypes."""
        for backend in available_backends():
            for dtype in INTEGER_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([0, 1, 4, 9], dtype=dtype))
                    self.assertIs(result.dtype, ts.float64)
                    self.assertEqual(result.tolist(), [0.0, 1.0, 2.0, 3.0])

    def test_a_negative_integer_gives_float64_nan(self):
        """Section 1.4: the integer converts first, then follows the rule."""
        for backend in available_backends():
            for dtype in (ts.int64, ts.int32, ts.int16, ts.int8):
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([-4], dtype=dtype))
                    self.assertIs(result.dtype, ts.float64)
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_an_int64_beyond_float64_follows_the_converted_operand(self):
        """Section 1.1: the conversion rule, stated as a measurement."""
        operand = 2**62 + 1
        # The specification roots the converted operand, and float64 rounds
        # 2**62 + 1 to 2**62, whose root is exactly 2**31.
        expected = reference_sqrt(float(operand), "float64")
        self.assertEqual(expected, 2147483648.0)
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sqrt(ts.Tensor([operand], dtype=ts.int64))
                self.assertEqual(result.tolist(), [expected])

    def test_both_signed_zeros_keep_their_sign(self):
        """Section 1.3: sqrt(-0.0) is -0.0, unlike abs and sign."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([0.0, -0.0], dtype=dtype))
                    produced = result.tolist()
                    self.assertEqual(produced, [0.0, 0.0])
                    self.assertEqual(math.copysign(1.0, produced[0]), 1.0)
                    self.assertEqual(
                        math.copysign(1.0, produced[1]),
                        -1.0,
                        "sqrt must not canonicalise negative zero",
                    )

    def test_negative_values_give_nan_rather_than_raising(self):
        """Section 1.4: a negative operand is a value, not a domain error."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(
                            ts.Tensor([-1.0, -4.0, -math.inf], dtype=dtype)
                        )
                    for item in result.tolist():
                        self.assertTrue(math.isnan(item))

    def test_one_negative_element_does_not_deny_the_others_a_result(self):
        """Section 1.4: the rule stays elementwise."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    produced = ts.sqrt(
                        ts.Tensor([4.0, -1.0, 9.0], dtype=ts.float64)
                    ).tolist()
                self.assertEqual(produced[0], 2.0)
                self.assertTrue(math.isnan(produced[1]))
                self.assertEqual(produced[2], 3.0)

    def test_positive_infinity_is_preserved(self):
        """Section 1.5."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([math.inf], dtype=dtype))
                    self.assertEqual(result.tolist(), [math.inf])

    def test_nan_is_classified_as_nan_only(self):
        """Section 1.6: assert the classification, never a payload or sign."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([math.nan], dtype=dtype))
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_float32_subnormal_operands_are_not_flushed(self):
        """Section 1.7, against the independent reference."""
        operands = [SMALLEST_FLOAT32_SUBNORMAL, LARGEST_FLOAT32_SUBNORMAL]
        expected = [reference_sqrt(value, "float32") for value in operands]
        # The specification names this one explicitly.
        self.assertEqual(expected[0], 3.743392066509216e-23)
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sqrt(ts.Tensor(operands, dtype=ts.float32))
                self.assertEqual(result.tolist(), expected)

    def test_float64_subnormal_operands_are_not_flushed(self):
        """Section 1.7 at binary64."""
        operands = [SMALLEST_FLOAT64_SUBNORMAL]
        expected = [reference_sqrt(value, "float64") for value in operands]
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sqrt(ts.Tensor(operands, dtype=ts.float64))
                self.assertEqual(result.tolist(), expected)


class SqrtCorrectRoundingTests(unittest.TestCase):
    """Section 1.2, against the independent reference of section 5."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _assert_correctly_rounded(self, backend, operands, dtype, reference_name):
        expected = [reference_sqrt(value, reference_name) for value in operands]
        with ts.use_backend(backend):
            produced = ts.sqrt(ts.Tensor(operands, dtype=dtype)).tolist()
        for index, (value, want, got) in enumerate(zip(operands, expected, produced)):
            if want != got:
                self.fail(
                    f"{backend} sqrt({value!r}) in {dtype.name}: "
                    f"expected {want!r}, produced {got!r} (element {index})"
                )

    def test_float32_is_correctly_rounded_across_the_exponent_range(self):
        operands = float32_sample(512, seed=20260921)
        for backend in available_backends():
            with self.subTest(backend=backend):
                self._assert_correctly_rounded(backend, operands, ts.float32, "float32")

    def test_float32_is_correctly_rounded_at_rounding_boundaries(self):
        """Operands whose true roots sit as close as possible to a boundary.

        These are the cases a square root computed in a wider format and then
        rounded down would get wrong if binary64 did not carry enough bits.
        """
        from fractions import Fraction

        from tests.operations.arithmetic._reference import round_fraction

        operands = []
        # Squaring overflows to infinity or underflows to zero for part of
        # the exponent range, so more are drawn than are kept.
        for value in float32_sample(1024, seed=5150):
            lower = Fraction(*float(value).as_integer_ratio())
            step = Fraction(*float(math.ulp(value)).as_integer_ratio())
            midpoint = lower + step / 2
            candidate = round_fraction(midpoint * midpoint, "float32")
            if math.isfinite(candidate) and candidate > 0.0:
                operands.append(candidate)
        self.assertGreater(len(operands), 400)
        for backend in available_backends():
            with self.subTest(backend=backend):
                self._assert_correctly_rounded(backend, operands, ts.float32, "float32")

    def test_float64_is_correctly_rounded(self):
        operands = [
            2.0,
            3.0,
            5.0,
            1e-300,
            1e300,
            0.1,
            math.pi,
            SMALLEST_FLOAT64_SUBNORMAL,
        ]
        for backend in available_backends():
            with self.subTest(backend=backend):
                self._assert_correctly_rounded(backend, operands, ts.float64, "float64")


class SqrtExecutesOnTheSelectedBackendTests(unittest.TestCase):
    """Section 2: selection is an execution requirement, with no fallback."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _accelerated_backends(self):
        return [
            backend
            for backend in ts.available_backends()
            if backend in ("numpy", "cuda")
        ]

    def test_results_are_resident_on_the_selected_backend(self):
        for backend in available_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([4, 9], dtype=dtype))
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_a_one_element_tensor_stays_on_the_selected_backend(self):
        """No workload threshold sends small work to the Python reference."""
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sqrt(ts.Tensor([4], dtype=dtype))
                    self.assertEqual(result.tolist(), [2.0])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_sqrt_kernel(self):
        module = importlib.import_module(PYTHON_SQRT_MODULE)
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "sqrt",
                        side_effect=AssertionError(
                            "sqrt must run on the selected backend"
                        ),
                    ):
                        with ts.use_backend(backend):
                            result = ts.sqrt(ts.Tensor([4, 9], dtype=dtype))
                    self.assertEqual(result.tolist(), [2.0, 3.0])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_dispatcher_carries_no_threshold_fallback_or_domain_check(self):
        """Section 1.4 and section 2, read off the dispatch module."""
        import inspect

        from tensors.backend.dispatch.elementwise import sqrt as dispatcher

        source = inspect.getsource(dispatcher)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "tensors.backend.policy",
            "python.kernels",
            "as reference",
            "astype",
            # Section 1.4: nothing inspects operand values.
            "any(",
            "< 0",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)

    def test_no_kernel_rejects_a_negative_operand(self):
        """The old ValueError must be gone from all three kernels."""
        import inspect

        for module_name in (
            "tensors.backend.python.kernels.elementwise.sqrt",
            "tensors.backend.numpy.kernels.elementwise.sqrt",
            "tensors.backend.cuda.kernels.elementwise.sqrt",
        ):
            with self.subTest(module=module_name):
                source = inspect.getsource(importlib.import_module(module_name))
                self.assertNotIn("only defined for non-negative", source)

    def test_the_kernel_receives_native_values_rather_than_a_tensor(self):
        """Section 3: a kernel never sees a Tensor or its metadata."""
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.sqrt

        def spy(values, **keywords):
            seen["values"] = values
            return original(values, **keywords)

        with patch.object(numpy_backend, "sqrt", spy):
            with ts.use_backend("numpy"):
                ts.sqrt(ts.Tensor([[1, 4], [9, 16]], dtype=ts.int64))

        lowered = seen["values"]
        self.assertNotIsInstance(lowered, ts.Tensor)
        self.assertIsInstance(lowered, numpy.ndarray)
        self.assertEqual(lowered.shape, (2, 2))
        # Section 3: the operand arrives in the Tensor's own dtype, and the
        # kernel performs the conversion to float64 itself.
        self.assertEqual(lowered.dtype, numpy.dtype("int64"))


class SqrtFusionTests(unittest.TestCase):
    """Section 4: fused execution observably agrees with eager execution."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    @staticmethod
    def _fused_forward(values, wrapper):
        from tensors.graph.state import reset_graph_state

        import tensors.backend.cuda.kernels as cuda_backend

        reset_graph_state()
        try:
            with ts.use_backend("cuda"):
                variable = ts.Variable(
                    ts.Tensor(values, dtype=ts.float32), requires_grad=False
                )
                output = ts.sqrt(variable) * 1.0
                computation = ts.graph.Computation(output)
                variable.data = ts.Tensor(values, dtype=ts.float32)
                state = {}
                original = cuda_backend.fused_elementwise

                def spy(*arguments, **keywords):
                    produced = original(*arguments, **keywords)
                    state["declined"] = produced is None
                    state["steps"] = arguments[1]
                    return produced

                with patch.object(cuda_backend, "fused_elementwise", spy):
                    result = computation.forward()
                return result.tolist(), state
        finally:
            reset_graph_state()

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
    def test_a_fused_chain_matches_eager_and_the_specification(self):
        """The operands the fused kernel runs on, fused for real."""
        probe = [
            SMALLEST_FLOAT32_SUBNORMAL,
            0.0,
            -0.0,
            4.0,
            0.25,
            math.inf,
            math.nan,
        ]
        expected = [
            reference_sqrt(SMALLEST_FLOAT32_SUBNORMAL, "float32"),
            0.0,
            -0.0,
            2.0,
            0.5,
            math.inf,
            math.nan,
        ]
        count = 8_192
        values = (probe * ((count // len(probe)) + 1))[:count]

        with ts.use_backend("cuda"):
            eager = ts.sqrt(ts.Tensor(values, dtype=ts.float32)).tolist()
        fused, state = self._fused_forward(values, None)

        self.assertIn(("sqrt", None, False, None), state["steps"])
        self.assertFalse(
            state["declined"], "the fused kernel should run on these operands"
        )

        def canonical(item):
            return (
                "nan"
                if isinstance(item, float) and math.isnan(item)
                else (signed_zero(item))
            )

        self.assertEqual(
            [canonical(item) for item in fused],
            [canonical(item) for item in eager],
        )
        self.assertEqual(
            [canonical(item) for item in fused[: len(probe)]],
            [canonical(item) for item in expected],
        )

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
    def test_a_negative_operand_still_produces_the_specified_result(self):
        """Section 4: the fused attempt declines, and eager supplies the NaN."""
        probe = [4.0, -1.0, 9.0, -math.inf]
        count = 8_192
        values = (probe * ((count // len(probe)) + 1))[:count]

        fused, state = self._fused_forward(values, None)
        self.assertTrue(
            state["declined"],
            "the fused kernel still carries the older negative-operand guard",
        )
        self.assertEqual(fused[0], 2.0)
        self.assertTrue(math.isnan(fused[1]))
        self.assertEqual(fused[2], 3.0)
        self.assertTrue(math.isnan(fused[3]))


if __name__ == "__main__":
    unittest.main()
