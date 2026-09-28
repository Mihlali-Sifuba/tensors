"""The sign indicator and its subgradient.

The forward tests below are written against `docs/sign-semantics.md`. Every
expectation is a literal taken from that document's value table: none of them
is obtained by running a backend and recording what it returned, because no
backend is the reference for `sign`.

The differentiation tests at the bottom are unchanged and out of that
specification's scope.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.state import reset_graph_state

#: The smallest positive binary32 subnormal.
SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45

#: The smallest positive binary64 subnormal.
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

FLOAT_DTYPES = (ts.float64, ts.float32)
SIGNED_INTEGER_DTYPES = (ts.int64, ts.int32, ts.int16, ts.int8)
INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + (ts.uint8,)
ALL_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES

PYTHON_SIGN_MODULE = "tensors.backend.python.kernels.elementwise.sign"


def available_backends():
    return ts.available_backends()


def is_positive_zero(value):
    """Whether a value is zero with a clear sign bit."""
    return value == 0.0 and math.copysign(1.0, value) == 1.0


def canonical(value):
    """A comparable form that distinguishes NaN and the two signed zeros."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    return value


def representable_operands(dtype):
    """Operands covering the rule, within what ``dtype`` can hold.

    ``uint8`` cannot represent a negative operand at all, so it gets the
    non-negative half of the table rather than an unrepresentable literal.
    """
    if dtype is ts.uint8:
        return [0, 5], [0, 1]
    return [-2, 0, 5], [-1, 0, 1]


class SignSpecificationTests(unittest.TestCase):
    """Forward `sign` against docs/sign-semantics.md, on every backend."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_ordinary_values_classify_in_every_float_dtype(self):
        """Section 1.2: negative, zero and positive ordinary values."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sign(ts.Tensor([-3.5, 0.0, 4.25], dtype=dtype))
                    self.assertEqual(result.tolist(), [-1.0, 0.0, 1.0])

    def test_ordinary_values_classify_in_every_signed_integer_dtype(self):
        """Section 1.6: the same rule in the integer's own width."""
        for backend in available_backends():
            for dtype in SIGNED_INTEGER_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sign(ts.Tensor([-3, 0, 4], dtype=dtype))
                    self.assertEqual(result.tolist(), [-1, 0, 1])

    def test_uint8_yields_only_zero_and_one(self):
        """Section 1.6: uint8 is unsigned, so -1 is unreachable."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sign(ts.Tensor([0, 1, 255], dtype=ts.uint8))
                self.assertEqual(result.tolist(), [0, 1, 1])

    def test_dtype_and_shape_are_preserved_for_every_dtype(self):
        """Section 1.1: shape and dtype are unchanged, and storage is resident."""
        for backend in available_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Tensor([[1, 0], [2, 3]], dtype=dtype)
                        result = ts.sign(value)
                    self.assertIs(result.dtype, dtype)
                    self.assertEqual(result.shape, value.shape)
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_both_signed_zeros_give_canonical_positive_zero(self):
        """Section 1.3: -0.0 does not survive into the result."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sign(ts.Tensor([0.0, -0.0], dtype=dtype))
                    produced = result.tolist()
                    self.assertEqual(produced, [0.0, 0.0])
                    for item in produced:
                        self.assertTrue(is_positive_zero(item))

    def test_infinities_classify_by_their_sign(self):
        """Section 1.2: +inf is +1 and -inf is -1."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sign(ts.Tensor([math.inf, -math.inf], dtype=dtype))
                    self.assertEqual(result.tolist(), [1.0, -1.0])

    def test_nan_is_classified_as_nan_only(self):
        """Section 1.4: assert the classification, never a payload or sign."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.sign(ts.Tensor([math.nan], dtype=dtype))
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_float32_subnormals_are_finite_nonzero_values(self):
        """Section 1.5: the smallest binary32 subnormals are +1 and -1."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sign(ts.Tensor([smallest, -smallest], dtype=ts.float32))
                self.assertEqual(result.tolist(), [1.0, -1.0])

    def test_float64_subnormals_are_finite_nonzero_values(self):
        """Section 1.5: the same rule at binary64."""
        smallest = SMALLEST_FLOAT64_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sign(ts.Tensor([smallest, -smallest], dtype=ts.float64))
                self.assertEqual(result.tolist(), [1.0, -1.0])

    def test_large_int64_values_keep_their_classification(self):
        """Section 1.6: no integer operand is routed through float64."""
        big = 2**62 + 1
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.sign(ts.Tensor([big, -big], dtype=ts.int64))
                self.assertEqual(result.tolist(), [1, -1])


class SignExecutesOnTheSelectedBackendTests(unittest.TestCase):
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
                        result = ts.sign(ts.Tensor([3], dtype=dtype))
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_sign_kernel(self):
        """The Python kernel is not reachable from an accelerated selection."""
        module = importlib.import_module(PYTHON_SIGN_MODULE)
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                operands, expected = representable_operands(dtype)
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "sign",
                        side_effect=AssertionError(
                            "sign must run on the selected backend"
                        ),
                    ):
                        with ts.use_backend(backend):
                            result = ts.sign(ts.Tensor(operands, dtype=dtype))
                    self.assertEqual(result.tolist(), expected)
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_an_integer_dtype_is_not_handed_to_the_python_reference(self):
        """Integer sign is executed by the selected backend, not declined."""
        module = importlib.import_module(PYTHON_SIGN_MODULE)
        for backend in self._accelerated_backends():
            for dtype in INTEGER_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "sign",
                        side_effect=AssertionError("integer sign must not fall back"),
                    ):
                        with ts.use_backend(backend):
                            result = ts.sign(ts.Tensor([0, 7], dtype=dtype))
                    self.assertEqual(result.tolist(), [0, 1])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_kernel_receives_native_values_rather_than_a_tensor(self):
        """Section 3: a kernel never sees a Tensor or its metadata."""
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.sign

        def spy(values, **keywords):
            seen["values"] = values
            return original(values, **keywords)

        with patch.object(numpy_backend, "sign", spy):
            with ts.use_backend("numpy"):
                ts.sign(ts.Tensor([[-1, 2], [3, -4]], dtype=ts.int64))

        lowered = seen["values"]
        self.assertNotIsInstance(lowered, ts.Tensor)
        self.assertIsInstance(lowered, numpy.ndarray)
        self.assertEqual(lowered.shape, (2, 2))
        # Section 1.6: the operand arrives in its own width, not as float64.
        self.assertEqual(lowered.dtype, numpy.dtype("int64"))

    def test_the_dispatcher_carries_no_threshold_or_reference_fallback(self):
        """The removed machinery must not reappear in the dispatch module."""
        import inspect

        from tensors.backend.dispatch.elementwise import sign as dispatcher

        source = inspect.getsource(dispatcher)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "tensors.backend.policy",
            # The Python reference kernel is no longer reachable from here.
            "python.kernels",
            "as reference",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)


class SignFusionTests(unittest.TestCase):
    """Section 4: a fused sign equals an eager sign, exactly."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
    def test_fused_sign_matches_eager_sign_on_every_specified_class(self):
        import tensors.backend.cuda.kernels as cuda_backend

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
        # Above the fusion work threshold, so the planner actually fuses.
        count = 8_192
        values = (probe * ((count // len(probe)) + 1))[:count]

        with ts.use_backend("cuda"):
            eager = ts.sign(ts.Tensor(values, dtype=ts.float32)).tolist()

            variable = ts.Variable(
                ts.Tensor(values, dtype=ts.float32), requires_grad=False
            )
            output = ts.sign(variable) * 1.0
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

        # One comparison over canonical forms: NaN compares equal to NaN, and
        # a zero carries its sign into the comparison so +0.0 and -0.0 differ.
        self.assertEqual(
            [canonical(item) for item in produced],
            [canonical(item) for item in eager],
        )
        # Both paths must also satisfy the specification outright, so that a
        # shared defect cannot make them agree.
        self.assertEqual(
            [canonical(item) for item in produced[: len(probe)]],
            [
                canonical(value)
                for value in (1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, math.nan)
            ],
        )


class SignTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_sign_returns_negative_zero_and_positive_indicators(self):
        value = ts.Tensor([-3, 0, 4], dtype=ts.int32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1, 0, 1])
        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.int32)

    def test_sign_preserves_float_dtype(self):
        value = ts.Tensor([-3.0, 0.0, 4.0], dtype=ts.float32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1.0, 0.0, 1.0])
        self.assertIs(result.dtype, ts.float32)

    def test_sign_propagates_nan_to_value_and_first_gradient(self):
        value = ts.Variable([math.nan])

        result = ts.sign(value)
        gradient = ts.grad(result, value)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(gradient.item()))

    def test_sign_derivative_is_zero_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        gradient = ts.grad(ts.sign(value), value)

        self.assertEqual(gradient.tolist(), [0.0, 0.0])

    def test_sign_derivative_rejects_zero(self):
        value = ts.Variable([0.0])

        with self.assertRaisesRegex(ValueError, "undefined at zero"):
            ts.grad(ts.sign(value), value)

    def test_sign_has_zero_higher_derivatives_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        first = ts.grad(ts.sign(value), value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [0.0, 0.0])
        self.assertEqual(second.tolist(), [0.0, 0.0])

    def test_sign_passes_gradcheck_away_from_zero(self):
        self.assertTrue(ts.gradcheck(ts.sign, ts.Tensor([-2.0, 3.0])))

    def test_public_math_namespace_exposes_sign_function_and_class(self):
        self.assertEqual(ts.math.sign([-1.0, 0.0, 1.0]).tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(
            ts.math.Sign().forward(ts.Tensor([-1.0, 0.0, 1.0])).tolist(),
            [-1.0, 0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
