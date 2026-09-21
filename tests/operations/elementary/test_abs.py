"""Absolute value and its subgradient at zero.

The forward tests below are written against `docs/abs-semantics.md`. Every
expectation is a literal taken from that document's tables: none of them is
obtained by running a backend and recording what it returned, because no
backend is the reference for `abs`.

The differentiation tests at the bottom are unchanged and out of that
specification's scope.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts

#: The smallest positive binary32 subnormal.
SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45

#: The smallest positive binary64 subnormal.
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

FLOAT_DTYPES = (ts.float64, ts.float32)
SIGNED_INTEGER_DTYPES = (ts.int64, ts.int32, ts.int16, ts.int8)
INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + (ts.uint8,)
ALL_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES

#: Section 1.7: each signed dtype's least value and its unrepresentable
#: magnitude, written out rather than computed from the implementation.
SIGNED_MINIMA = (
    (ts.int8, -128),
    (ts.int16, -32768),
    (ts.int32, -2147483648),
    (ts.int64, -9223372036854775808),
)

PYTHON_ABS_MODULE = "tensors.backend.python.kernels.elementwise.abs"


def available_backends():
    return ts.available_backends()


def is_positive_zero(value):
    """Whether a value is zero with a clear sign bit."""
    return value == 0.0 and math.copysign(1.0, value) == 1.0


class AbsSpecificationTests(unittest.TestCase):
    """Forward `abs` against docs/abs-semantics.md, on every backend."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_ordinary_values_take_their_magnitude_in_every_float_dtype(self):
        """Section 1.2: negative, zero and positive ordinary values."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([-3.5, 0.0, 4.25], dtype=dtype))
                    self.assertEqual(result.tolist(), [3.5, 0.0, 4.25])

    def test_valid_signed_integers_take_their_exact_magnitude(self):
        """Section 1.2, including the values adjacent to each minimum."""
        for backend in available_backends():
            for dtype, minimum in SIGNED_MINIMA:
                operands = [minimum + 1, minimum + 2, -1, 0, 1, -minimum - 1]
                expected = [-(minimum + 1), -(minimum + 2), 1, 0, 1, -minimum - 1]
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor(operands, dtype=dtype))
                    self.assertEqual(result.tolist(), expected)
                    self.assertIs(result.dtype, dtype)

    def test_uint8_is_the_identity(self):
        """Section 1.2: uint8 is unsigned, so abs cannot change a value."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.abs(ts.Tensor([0, 1, 128, 255], dtype=ts.uint8))
                self.assertEqual(result.tolist(), [0, 1, 128, 255])

    def test_dtype_and_shape_are_preserved_for_every_dtype(self):
        """Section 1.1: shape and dtype unchanged, storage on the selection."""
        for backend in available_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Tensor([[1, 0], [2, 3]], dtype=dtype)
                        result = ts.abs(value)
                    self.assertIs(result.dtype, dtype)
                    self.assertEqual(result.shape, value.shape)
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_both_signed_zeros_give_canonical_positive_zero(self):
        """Section 1.3: -0.0 does not survive into the result."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([0.0, -0.0], dtype=dtype))
                    produced = result.tolist()
                    self.assertEqual(produced, [0.0, 0.0])
                    for item in produced:
                        self.assertTrue(is_positive_zero(item))

    def test_both_infinities_give_positive_infinity(self):
        """Section 1.5: neither infinity is an error."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([math.inf, -math.inf], dtype=dtype))
                    self.assertEqual(result.tolist(), [math.inf, math.inf])

    def test_nan_is_classified_as_nan_only(self):
        """Section 1.4: assert the classification, never a payload or sign."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([math.nan], dtype=dtype))
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_float32_subnormals_keep_their_magnitude(self):
        """Section 1.6: preserved as an operand and as a result."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.abs(ts.Tensor([smallest, -smallest], dtype=ts.float32))
                self.assertEqual(result.tolist(), [smallest, smallest])

    def test_float64_subnormals_keep_their_magnitude(self):
        """Section 1.6: the same rule at binary64."""
        smallest = SMALLEST_FLOAT64_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.abs(ts.Tensor([smallest, -smallest], dtype=ts.float64))
                self.assertEqual(result.tolist(), [smallest, smallest])


class AbsSignedMinimumTests(unittest.TestCase):
    """Section 1.7: the one input abs refuses, identically everywhere."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_each_signed_minimum_raises_the_specified_error(self):
        for backend in available_backends():
            for dtype, minimum in SIGNED_MINIMA:
                expected = f"abs({minimum}) is not representable in {dtype.name}"
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Tensor([minimum], dtype=dtype)
                        with self.assertRaises(OverflowError) as caught:
                            ts.abs(value)
                    self.assertEqual(str(caught.exception), expected)

    def test_the_minimum_is_refused_among_representable_neighbours(self):
        """One offending element is enough, wherever it sits."""
        for backend in available_backends():
            for dtype, minimum in SIGNED_MINIMA:
                expected = f"abs({minimum}) is not representable in {dtype.name}"
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Tensor([-1, 0, 1, minimum, 2], dtype=dtype)
                        with self.assertRaises(OverflowError) as caught:
                            ts.abs(value)
                    self.assertEqual(str(caught.exception), expected)

    def test_the_value_above_each_minimum_is_accepted(self):
        """The rule is exactly the least value, not a range near it."""
        for backend in available_backends():
            for dtype, minimum in SIGNED_MINIMA:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([minimum + 1], dtype=dtype))
                    self.assertEqual(result.tolist(), [-(minimum + 1)])

    def test_unsigned_and_floating_dtypes_never_raise(self):
        """uint8 has no negative value, and floating dtypes are symmetric."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    self.assertEqual(
                        ts.abs(ts.Tensor([0, 255], dtype=ts.uint8)).tolist(),
                        [0, 255],
                    )
                    self.assertEqual(
                        ts.abs(ts.Tensor([-1.5], dtype=ts.float64)).tolist(),
                        [1.5],
                    )


class AbsExecutesOnTheSelectedBackendTests(unittest.TestCase):
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

    def test_a_one_element_tensor_stays_on_the_selected_backend(self):
        """No workload threshold sends small work to the Python reference."""
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([3], dtype=dtype))
                    self.assertEqual(result.tolist(), [3])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_integer_results_are_no_longer_python_storage(self):
        """Integer abs used to decline and return Python storage."""
        for backend in self._accelerated_backends():
            for dtype in INTEGER_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.abs(ts.Tensor([0, 7], dtype=dtype))
                    self.assertEqual(result.tolist(), [0, 7])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_abs_kernel(self):
        """The Python kernel is not reachable from an accelerated selection."""
        module = importlib.import_module(PYTHON_ABS_MODULE)
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "abs",
                        side_effect=AssertionError(
                            "abs must run on the selected backend"
                        ),
                    ):
                        with ts.use_backend(backend):
                            result = ts.abs(ts.Tensor([2, 0, 5], dtype=dtype))
                    self.assertEqual(result.tolist(), [2, 0, 5])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_signed_minimum_is_refused_without_the_python_kernel(self):
        """The precheck happens in the dispatcher, not in a reference kernel."""
        module = importlib.import_module(PYTHON_ABS_MODULE)
        for backend in self._accelerated_backends():
            for dtype, minimum in SIGNED_MINIMA:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "abs",
                        side_effect=AssertionError("no reference kernel"),
                    ):
                        with ts.use_backend(backend):
                            with self.assertRaises(OverflowError):
                                ts.abs(ts.Tensor([minimum], dtype=dtype))

    def test_the_kernel_receives_native_values_rather_than_a_tensor(self):
        """Section 3: a kernel never sees a Tensor or its metadata."""
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.abs

        def spy(values, **keywords):
            seen["values"] = values
            return original(values, **keywords)

        with patch.object(numpy_backend, "abs", spy):
            with ts.use_backend("numpy"):
                ts.abs(ts.Tensor([[-1, 2], [3, -4]], dtype=ts.int64))

        lowered = seen["values"]
        self.assertNotIsInstance(lowered, ts.Tensor)
        self.assertIsInstance(lowered, numpy.ndarray)
        self.assertEqual(lowered.shape, (2, 2))
        # Section 1.1: the operand arrives in its own width, not as float64.
        self.assertEqual(lowered.dtype, numpy.dtype("int64"))

    def test_the_dispatcher_carries_no_threshold_or_reference_fallback(self):
        """The removed machinery must not reappear in the dispatch module."""
        import inspect

        from tensors.backend.dispatch.elementwise import abs as dispatcher

        source = inspect.getsource(dispatcher)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "tensors.backend.policy",
            "python.kernels",
            "as reference",
            "astype",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)


class AbsFusionTests(unittest.TestCase):
    """Section 4: a fused abs equals an eager abs, exactly."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
    def test_fused_abs_matches_eager_abs_on_every_specified_class(self):
        from tensors.graph.state import reset_graph_state

        import tensors.backend.cuda.kernels as cuda_backend

        reset_graph_state()
        self.addCleanup(reset_graph_state)

        smallest = SMALLEST_FLOAT32_SUBNORMAL
        probe = [
            smallest,
            -smallest,
            0.0,
            -0.0,
            3.5,
            -3.5,
            math.inf,
            -math.inf,
            math.nan,
        ]
        # Section 1.2/1.5/1.6 read literally, in probe order.
        specified = [
            smallest,
            smallest,
            0.0,
            0.0,
            3.5,
            3.5,
            math.inf,
            math.inf,
            math.nan,
        ]
        # Above the fusion work threshold, so the planner actually fuses.
        count = 8_192
        values = (probe * ((count // len(probe)) + 1))[:count]

        def canonical(item):
            if isinstance(item, float) and math.isnan(item):
                return "nan"
            if item == 0:
                return "-0.0" if math.copysign(1.0, item) < 0 else "0.0"
            return item

        with ts.use_backend("cuda"):
            eager = ts.abs(ts.Tensor(values, dtype=ts.float32)).tolist()

            variable = ts.Variable(
                ts.Tensor(values, dtype=ts.float32), requires_grad=False
            )
            output = ts.abs(variable) * 1.0
            computation = ts.graph.Computation(output)
            variable.data = ts.Tensor(values, dtype=ts.float32)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fused:
                fused_result = computation.forward()

        fused.assert_called_once()
        produced = fused_result.tolist()

        self.assertEqual(
            [canonical(item) for item in produced],
            [canonical(item) for item in eager],
        )
        # Both paths must also satisfy the specification outright, so that a
        # shared defect cannot make them agree.
        self.assertEqual(
            [canonical(item) for item in produced[: len(probe)]],
            [canonical(item) for item in specified],
        )


class AbsoluteValueTests(unittest.TestCase):
    def test_abs_preserves_shape_and_dtype(self):
        value = ts.Tensor([[-2.0, 0.0], [3.0, -4.0]], dtype=ts.float32)

        result = abs(value)

        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [2.0, 0.0, 3.0, 4.0])

    def test_abs_uses_zero_subgradient_at_zero(self):
        value = ts.Variable([-2.0, 0.0, 3.0])

        first = ts.grad(ts.sum(ts.abs(value)), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(second.tolist(), [0.0, 0.0, 0.0])

    def test_abs_propagates_nan_to_value_and_gradient(self):
        value = ts.Variable([math.nan])
        result = ts.abs(value)

        ts.backward(result)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(value.grad.item()))

    def test_abs_rejects_higher_derivative_at_nan(self):
        value = ts.Variable([math.nan])

        with self.assertRaisesRegex(ValueError, "undefined at NaN"):
            ts.grad(ts.abs(value), value, create_graph=True)


if __name__ == "__main__":
    unittest.main()
