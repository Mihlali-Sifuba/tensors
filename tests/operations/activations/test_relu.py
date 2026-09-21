"""Forward ReLU, against docs/relu-semantics.md.

Every expectation is a literal from that document's tables. None is obtained
by running a backend and recording what it returned, and the fused path is
checked against the specification as well as against the eager path, so a
defect shared by both cannot pass.

Differentiation is outside this specification and is covered by
`test_activations.py`.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts

#: The smallest positive binary32 subnormal.
SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45

#: The largest binary32 subnormal.
LARGEST_FLOAT32_SUBNORMAL = 1.1754942106924411e-38

#: The smallest positive binary64 subnormal.
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

FLOAT_DTYPES = (ts.float64, ts.float32)
SIGNED_INTEGER_DTYPES = (ts.int64, ts.int32, ts.int16, ts.int8)
INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + (ts.uint8,)
ALL_DTYPES = FLOAT_DTYPES + INTEGER_DTYPES

PYTHON_RELU_MODULE = "tensors.backend.python.kernels.elementwise.relu"


def available_backends():
    return ts.available_backends()


def canonical(value):
    """A comparable form separating NaN and the two signed zeros."""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    return value


def is_positive_zero(value):
    return value == 0.0 and math.copysign(1.0, value) == 1.0


class ReluSpecificationTests(unittest.TestCase):
    """Section 1, on every backend."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_ordinary_values_on_every_float_dtype(self):
        """Section 1.2: positive kept, negative and zero rectified."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(
                            ts.Tensor([-3.5, 0.0, 2.25, -0.125], dtype=dtype)
                        )
                    self.assertEqual(result.tolist(), [0.0, 0.0, 2.25, 0.0])

    def test_ordinary_values_on_every_signed_integer_dtype(self):
        """Section 1.2 in the integer's own width."""
        for backend in available_backends():
            for dtype in SIGNED_INTEGER_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([-7, -1, 0, 1, 9], dtype=dtype))
                    self.assertEqual(result.tolist(), [0, 0, 0, 1, 9])

    def test_uint8_is_the_identity(self):
        """Section 1.2: every uint8 value is non-negative."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.relu(ts.Tensor([0, 1, 128, 255], dtype=ts.uint8))
                self.assertEqual(result.tolist(), [0, 1, 128, 255])

    def test_large_int64_values_are_not_routed_through_float64(self):
        """Section 1.2: a value above 2**53 comes back exactly."""
        big = 2**62 + 1
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.relu(ts.Tensor([big, -big], dtype=ts.int64))
                self.assertEqual(result.tolist(), [big, 0])
                self.assertIs(result.dtype, ts.int64)

    def test_dtype_and_shape_are_preserved_for_every_dtype(self):
        """Section 1.1, over scalar, flat and multidimensional shapes."""
        shapes = ([3], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]])
        for backend in available_backends():
            for dtype in ALL_DTYPES:
                for data in shapes:
                    with self.subTest(backend=backend, dtype=dtype.name, data=data):
                        with ts.use_backend(backend):
                            value = ts.Tensor(data, dtype=dtype)
                            result = ts.relu(value)
                        self.assertIs(result.dtype, dtype)
                        self.assertEqual(result.shape, value.shape)

    def test_a_one_element_tensor_is_rectified(self):
        """Section 1.2 at the smallest shape that exists."""
        for backend in available_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        self.assertEqual(
                            ts.relu(ts.Tensor([5], dtype=dtype)).tolist(), [5]
                        )

    def test_both_signed_zeros_give_canonical_positive_zero(self):
        """Section 1.3: -0.0 does not survive into the result."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([0.0, -0.0], dtype=dtype))
                    produced = result.tolist()
                    self.assertEqual(produced, [0.0, 0.0])
                    for item in produced:
                        self.assertTrue(
                            is_positive_zero(item),
                            "ReLU must canonicalise both zeros to +0.0",
                        )

    def test_a_negative_value_produces_positive_zero(self):
        """Section 1.3: the zero the negative branch produces is positive."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([-2.5], dtype=dtype))
                    self.assertTrue(is_positive_zero(result.tolist()[0]))

    def test_infinities(self):
        """Section 1.2: +inf is kept and -inf rectifies to +0.0."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([math.inf, -math.inf], dtype=dtype))
                    produced = result.tolist()
                    self.assertEqual(produced[0], math.inf)
                    self.assertTrue(is_positive_zero(produced[1]))

    def test_nan_is_classified_as_nan_only(self):
        """Section 1.5: assert the classification, never a payload or sign."""
        for backend in available_backends():
            for dtype in FLOAT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([math.nan], dtype=dtype))
                    self.assertTrue(math.isnan(result.tolist()[0]))

    def test_positive_float32_subnormals_are_returned_unchanged(self):
        """Section 1.4: the smallest and largest, neither flushed."""
        operands = [SMALLEST_FLOAT32_SUBNORMAL, LARGEST_FLOAT32_SUBNORMAL]
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.relu(ts.Tensor(operands, dtype=ts.float32))
                self.assertEqual(result.tolist(), operands)

    def test_negative_float32_subnormals_give_positive_zero(self):
        """Section 1.4: a negative subnormal is just a negative value."""
        operands = [-SMALLEST_FLOAT32_SUBNORMAL, -LARGEST_FLOAT32_SUBNORMAL]
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.relu(ts.Tensor(operands, dtype=ts.float32))
                for item in result.tolist():
                    self.assertTrue(is_positive_zero(item))

    def test_float64_subnormals(self):
        """Section 1.4 at binary64, both signs."""
        smallest = SMALLEST_FLOAT64_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    result = ts.relu(ts.Tensor([smallest, -smallest], dtype=ts.float64))
                produced = result.tolist()
                self.assertEqual(produced[0], smallest)
                self.assertTrue(is_positive_zero(produced[1]))


class ReluExecutesOnTheSelectedBackendTests(unittest.TestCase):
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
                        result = ts.relu(ts.Tensor([1, 2], dtype=dtype))
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_a_one_element_tensor_stays_on_the_selected_backend(self):
        """No workload threshold sends small work to the Python reference."""
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        result = ts.relu(ts.Tensor([7], dtype=dtype))
                    self.assertEqual(result.tolist(), [7])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_relu_kernel(self):
        module = importlib.import_module(PYTHON_RELU_MODULE)
        for backend in self._accelerated_backends():
            for dtype in ALL_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "relu",
                        side_effect=AssertionError(
                            "relu must run on the selected backend"
                        ),
                    ):
                        with ts.use_backend(backend):
                            result = ts.relu(ts.Tensor([2, 0, 5], dtype=dtype))
                    self.assertEqual(result.tolist(), [2, 0, 5])
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_a_declining_kernel_is_reported_rather_than_worked_around(self):
        """Section 2: a decline raises instead of running somewhere else."""
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import tensors.backend.numpy.kernels as numpy_backend

        from tensors.backend.config import BackendOperationUnsupportedError

        with patch.object(numpy_backend, "relu", return_value=None):
            with ts.use_backend("numpy"):
                with self.assertRaises(BackendOperationUnsupportedError) as caught:
                    ts.relu(ts.Tensor([1.0], dtype=ts.float64))
        self.assertIn("numpy", str(caught.exception))
        self.assertIn("relu", str(caught.exception))

    def test_the_kernel_receives_native_values_rather_than_a_tensor(self):
        """Section 3: a kernel never sees a Tensor or its metadata."""
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.relu

        def spy(values, **keywords):
            seen["values"] = values
            return original(values, **keywords)

        with patch.object(numpy_backend, "relu", spy):
            with ts.use_backend("numpy"):
                ts.relu(ts.Tensor([[-1, 2], [3, -4]], dtype=ts.int64))

        lowered = seen["values"]
        self.assertNotIsInstance(lowered, ts.Tensor)
        self.assertIsInstance(lowered, numpy.ndarray)
        self.assertEqual(lowered.shape, (2, 2))
        # Section 1.1: the operand arrives in its own width, not as float64.
        self.assertEqual(lowered.dtype, numpy.dtype("int64"))

    def test_the_dispatcher_carries_no_threshold_fallback_or_inspection(self):
        """Sections 1.6 and 2, read off the dispatch module."""
        import inspect

        from tensors.backend.dispatch.elementwise import relu as dispatcher

        source = inspect.getsource(dispatcher)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "tensors.backend.policy",
            "python.kernels",
            "as reference",
            "astype",
            # Section 1.6: nothing inspects operand values.
            "any(",
            "> 0",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)


class ReluFusionTests(unittest.TestCase):
    """Section 4: fused execution agrees with eager and with the document."""

    #: Probe operands, and the results section 1.2 requires for them.
    PROBE = (
        SMALLEST_FLOAT32_SUBNORMAL,
        -SMALLEST_FLOAT32_SUBNORMAL,
        LARGEST_FLOAT32_SUBNORMAL,
        0.0,
        -0.0,
        -1.5,
        2.5,
        math.inf,
        -math.inf,
        math.nan,
    )
    SPECIFIED = (
        SMALLEST_FLOAT32_SUBNORMAL,
        0.0,
        LARGEST_FLOAT32_SUBNORMAL,
        0.0,
        0.0,
        0.0,
        2.5,
        math.inf,
        0.0,
        math.nan,
    )

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    @unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
    def test_fused_relu_matches_eager_and_the_specification(self):
        from tensors.graph.state import reset_graph_state

        import tensors.backend.cuda.kernels as cuda_backend

        reset_graph_state()
        self.addCleanup(reset_graph_state)

        count = 8_192
        probe = list(self.PROBE)
        values = (probe * ((count // len(probe)) + 1))[:count]

        state = {}
        with ts.use_backend("cuda"):
            eager = ts.relu(ts.Tensor(values, dtype=ts.float32)).tolist()

            variable = ts.Variable(
                ts.Tensor(values, dtype=ts.float32), requires_grad=False
            )
            output = ts.relu(variable) * 1.0
            computation = ts.graph.Computation(output)
            variable.data = ts.Tensor(values, dtype=ts.float32)

            original = cuda_backend.fused_elementwise

            def spy(*arguments, **keywords):
                produced = original(*arguments, **keywords)
                state["declined"] = produced is None
                state["steps"] = arguments[1]
                state["calls"] = state.get("calls", 0) + 1
                return produced

            with patch.object(cuda_backend, "fused_elementwise", spy):
                fused = computation.forward().tolist()

        # The plan really contains ReLU, the fused kernel really ran, and it
        # was not answered by a decline falling back to eager execution.
        self.assertIn(("relu", None, False, None), state["steps"])
        self.assertEqual(state.get("calls"), 1)
        self.assertFalse(
            state["declined"],
            "the fused kernel declined, so this comparison would be vacuous",
        )

        self.assertEqual(
            [canonical(item) for item in fused],
            [canonical(item) for item in eager],
        )
        # Both paths must satisfy the document outright, so that a shared
        # defect cannot make them agree.
        expected = [canonical(item) for item in self.SPECIFIED]
        self.assertEqual([canonical(item) for item in fused[: len(probe)]], expected)
        self.assertEqual([canonical(item) for item in eager[: len(probe)]], expected)


if __name__ == "__main__":
    unittest.main()
