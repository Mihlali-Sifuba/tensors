"""The ReLU VJP, against docs/relu-semantics.md sections 6 to 8.

Every expectation is a literal from those sections. No backend is used as
another backend's oracle, and the graph-built VJP is compared both against
the eager VJP and against the document.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.state import reset_graph_state

SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45
LARGEST_FLOAT32_SUBNORMAL = 1.1754942106924411e-38
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

GRADIENT_DTYPES = (ts.float64, ts.float32)

#: Section 6.1: upstreams that must not reach the inactive side's result.
HOSTILE_UPSTREAMS = (-3.0, math.inf, -math.inf, math.nan)

PYTHON_VJP_MODULE = "tensors.backend.python.kernels.elementwise.relu_gradient"


def available_backends():
    return ts.available_backends()


def is_positive_zero(value):
    return value == 0.0 and math.copysign(1.0, value) == 1.0


def vjp(backend, primals, upstreams, dtype, create_graph=False):
    with ts.use_backend(backend):
        value = ts.Variable(ts.Tensor(primals, dtype=dtype))
        produced = ts.grad(
            ts.relu(value),
            value,
            grad_outputs=ts.Tensor(upstreams, dtype=dtype),
            create_graph=create_graph,
        )
        return produced.data if create_graph else produced


class ReluVjpValueTests(unittest.TestCase):
    """Section 6: the routed first-order VJP."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_the_positive_side_passes_the_upstream_through(self):
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(backend, [2.0, 5.0], [3.0, -4.0], dtype).tolist()
                    self.assertEqual(produced, [3.0, -4.0])

    def test_the_inactive_side_gives_canonical_positive_zero(self):
        """Section 6: negative, both zeros and -inf are all inactive."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(
                        backend,
                        [-2.0, 0.0, -0.0, -math.inf],
                        [3.0, 3.0, 3.0, 3.0],
                        dtype,
                    ).tolist()
                    for item in produced:
                        self.assertTrue(is_positive_zero(item))

    def test_the_inactive_side_ignores_a_hostile_upstream(self):
        """Section 6.1: routing, not multiplication."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                for upstream in HOSTILE_UPSTREAMS:
                    with self.subTest(
                        backend=backend, dtype=dtype.name, upstream=upstream
                    ):
                        produced = vjp(
                            backend, [-1.0, 0.0], [upstream, upstream], dtype
                        ).tolist()
                        for item in produced:
                            self.assertTrue(
                                is_positive_zero(item),
                                f"upstream {upstream!r} leaked into the "
                                "inactive side",
                            )

    def test_the_active_side_keeps_an_infinite_or_nan_upstream(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend, [2.0, 2.0], [math.inf, math.nan], ts.float64
                ).tolist()
                self.assertEqual(produced[0], math.inf)
                self.assertTrue(math.isnan(produced[1]))

    def test_a_nan_primal_gives_nan(self):
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(backend, [math.nan], [3.0], dtype).tolist()
                    self.assertTrue(math.isnan(produced[0]))

    def test_positive_float32_subnormal_primals_are_active(self):
        """Section 6.2: the whole band, not just its lower edge."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        largest = LARGEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend, [smallest, largest], [3.0, 3.0], ts.float32
                ).tolist()
                self.assertEqual(produced, [3.0, 3.0])

    def test_negative_float32_subnormal_primals_are_inactive(self):
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        largest = LARGEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend, [-smallest, -largest], [3.0, 3.0], ts.float32
                ).tolist()
                for item in produced:
                    self.assertTrue(is_positive_zero(item))

    def test_float64_subnormal_primals_select_their_branch(self):
        smallest = SMALLEST_FLOAT64_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend, [smallest, -smallest], [3.0, 3.0], ts.float64
                ).tolist()
                self.assertEqual(produced[0], 3.0)
                self.assertTrue(is_positive_zero(produced[1]))

    def test_a_subnormal_upstream_passes_the_active_branch_unchanged(self):
        """Section 6.2: a selection must not flush what it selects."""
        for upstream in (SMALLEST_FLOAT32_SUBNORMAL, LARGEST_FLOAT32_SUBNORMAL):
            for backend in available_backends():
                with self.subTest(backend=backend, upstream=upstream):
                    produced = vjp(backend, [1.0], [upstream], ts.float32).tolist()
                    self.assertEqual(produced, [upstream])

    def test_no_input_raises(self):
        """Section 6: the ReLU VJP has no domain error."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend,
                    [0.0, -0.0, math.nan, math.inf, -math.inf],
                    [1.0] * 5,
                    ts.float64,
                ).tolist()
                self.assertTrue(is_positive_zero(produced[0]))
                self.assertTrue(is_positive_zero(produced[1]))
                self.assertTrue(math.isnan(produced[2]))
                self.assertEqual(produced[3], 1.0)
                self.assertTrue(is_positive_zero(produced[4]))

    def test_dtype_shape_and_residency(self):
        """Section 6.3."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(
                            ts.Tensor([[1.0, -2.0], [3.0, -4.0]], dtype=dtype)
                        )
                        produced = ts.grad(ts.relu(value), value)
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.shape, value.data.shape)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_mismatched_upstream_is_rejected(self):
        from tensors.operations.activations.relu import ReLU
        from tensors.operations.base import UNARY_DEMAND

        with ts.use_backend("python"):
            value = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "shape"):
                ReLU().backward(
                    ts.Tensor([1.0], dtype=ts.float64),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )
            with self.assertRaisesRegex(ValueError, "dtype"):
                ReLU().backward(
                    ts.Tensor([1.0, 2.0], dtype=ts.float32),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )


class ReluVjpExecutionTests(unittest.TestCase):
    """Section 6.3: strict selected-backend execution."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def _accelerated(self):
        return [b for b in ts.available_backends() if b in ("numpy", "cuda")]

    def test_a_one_element_workload_stays_on_the_selected_backend(self):
        for backend in self._accelerated():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(ts.Tensor([3.0], dtype=dtype))
                        produced = ts.grad(ts.relu(value), value)
                    self.assertEqual(produced.backend_storage.kind, backend)
                    self.assertEqual(produced.tolist(), [1.0])

    def test_the_selected_backend_does_not_use_the_python_kernel(self):
        module = importlib.import_module(PYTHON_VJP_MODULE)
        for backend in self._accelerated():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "relu_gradient",
                        side_effect=AssertionError("no Python fallback"),
                    ):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([3.0, -1.0], dtype=dtype))
                            produced = ts.grad(ts.relu(value), value)
                    self.assertEqual(produced.tolist(), [1.0, 0.0])
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_declining_kernel_is_reported(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import tensors.backend.numpy.kernels as numpy_backend

        from tensors.backend.config import BackendOperationUnsupportedError

        with patch.object(numpy_backend, "relu_gradient", return_value=None):
            with ts.use_backend("numpy"):
                value = ts.Variable(ts.Tensor([1.0], dtype=ts.float64))
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.grad(ts.relu(value), value)

    def test_the_kernel_receives_native_values_rather_than_tensors(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.relu_gradient

        def spy(grad_values, values, **keywords):
            seen["grad"] = grad_values
            seen["value"] = values
            return original(grad_values, values, **keywords)

        with patch.object(numpy_backend, "relu_gradient", spy):
            with ts.use_backend("numpy"):
                value = ts.Variable(
                    ts.Tensor([[1.0, -2.0], [3.0, -4.0]], dtype=ts.float64)
                )
                ts.grad(ts.relu(value), value)

        for name in ("grad", "value"):
            self.assertNotIsInstance(seen[name], ts.Tensor)
            self.assertIsInstance(seen[name], numpy.ndarray)
            self.assertEqual(seen[name].shape, (2, 2))

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.elementwise import relu_gradient as dispatcher

        source = inspect.getsource(dispatcher)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "tensors.backend.policy",
            "python.kernels",
            "as reference",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("BackendOperationUnsupportedError", source)


class ReluGraphVjpTests(unittest.TestCase):
    """Sections 7 and 8: the graph-built VJP and higher-order regions."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_the_graph_built_vjp_equals_the_eager_vjp(self):
        primals = [2.0, -3.0, 0.0, -0.0, math.nan]
        upstreams = [-4.0] * 5
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    eager = vjp(backend, primals, upstreams, dtype)
                    built = vjp(backend, primals, upstreams, dtype, create_graph=True)
                    self.assertEqual(
                        [math.isnan(x) for x in built.tolist()],
                        [math.isnan(x) for x in eager.tolist()],
                    )
                    for produced, expected in zip(built.tolist(), eager.tolist()):
                        if not math.isnan(expected):
                            self.assertEqual(produced, expected)
                            self.assertEqual(
                                math.copysign(1.0, produced),
                                math.copysign(1.0, expected),
                            )
                    self.assertIs(built.dtype, eager.dtype)
                    self.assertEqual(built.shape, eager.shape)
                    self.assertEqual(
                        built.backend_storage.kind, eager.backend_storage.kind
                    )

    def test_the_graph_built_inactive_side_ignores_a_hostile_upstream(self):
        for backend in available_backends():
            for upstream in HOSTILE_UPSTREAMS:
                with self.subTest(backend=backend, upstream=upstream):
                    built = vjp(
                        backend, [-1.0], [upstream], ts.float64, create_graph=True
                    ).tolist()
                    self.assertTrue(is_positive_zero(built[0]))

    def test_the_second_derivative_is_zero_away_from_the_kink(self):
        """Section 8."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([-2.0, 3.0]))
                    first = ts.grad(ts.relu(value), value, create_graph=True)
                    second = ts.grad(first, value, grad_outputs=ts.Tensor([1.0, 1.0]))
                self.assertEqual(second.tolist(), [0.0, 0.0])

    def test_the_second_derivative_raises_at_the_kink(self):
        """Section 8."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([0.0, 2.0]))
                    first = ts.grad(ts.relu(value), value, create_graph=True)
                    with self.assertRaisesRegex(
                        ValueError, "relu second derivative is undefined at zero"
                    ):
                        ts.grad(first, value, grad_outputs=ts.Tensor([1.0, 1.0]))

    def test_a_nan_primal_propagates_through_the_graph_vjp(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                built = vjp(
                    backend, [math.nan], [1.0], ts.float64, create_graph=True
                ).tolist()
                self.assertTrue(math.isnan(built[0]))

    def test_backward_graph_reads_no_host_values(self):
        """Section 7, checked structurally on the parsed statements."""
        import ast
        import inspect
        import textwrap

        module = importlib.import_module("tensors.operations.activations.relu")
        tree = ast.parse(textwrap.dedent(inspect.getsource(module.ReLU.backward_graph)))
        body = tree.body[0].body
        if (
            isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        comprehensions = []
        host_reads = []
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(
                    node,
                    (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp),
                ):
                    comprehensions.append(node)
                if isinstance(node, ast.Attribute) and node.attr == "_data":
                    host_reads.append(node)
        self.assertEqual(
            comprehensions, [], "backward_graph must not build a Python mask"
        )
        self.assertEqual(host_reads, [], "backward_graph must not read host values")

    def test_a_compiled_first_vjp_graph_replays_with_changed_values(self):
        """Section 7: the recorded VJP is re-executed, not re-derived.

        The gradient itself is compiled and replayed. An earlier version of
        this test compiled only the forward operation and then took a fresh
        gradient after each replacement; a fresh gradient re-enters
        `backward_graph`, so it would have passed even if the recorded
        graph had frozen its branch decision.
        """
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([2.0, 3.0], dtype=ts.float32))
                    first = ts.grad(ts.relu(value), value, create_graph=True)
                    computation = ts.graph.Computation(first)
                    self.assertEqual(computation.forward().tolist(), [1.0, 1.0])

                    expectations = (
                        ([-2.0, -3.0], [0.0, 0.0]),
                        ([smallest, -smallest], [1.0, 0.0]),
                        ([0.0, 4.0], [0.0, 1.0]),
                    )
                    for replacement, expected in expectations:
                        value.data = ts.Tensor(replacement, dtype=ts.float32)
                        produced = computation.forward()
                        self.assertEqual(produced.tolist(), expected)
                        self.assertIs(produced.dtype, ts.float32)
                        self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_compiled_second_derivative_graph_keeps_this_operations_error(self):
        """Section 8, through a replayed graph rather than a fresh one.

        This is the case that exposed the defect the internal
        `ReLUPrimalVJP` node fixes. The second derivative's primal partial
        borrows its numbers from the sign VJP, and recording the sign VJP
        directly made a *replayed* kink report ``sign derivative is
        undefined at zero``. The error has to be raised by the recorded
        operation's own forward, which is what this replays.
        """
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(ts.Tensor([2.0], dtype=dtype))
                        first = ts.grad(ts.relu(value), value, create_graph=True)
                        second = ts.grad(first, value, create_graph=True)
                        computation = ts.graph.Computation(second)

                        produced = computation.forward()
                        self.assertEqual(produced.tolist(), [0.0])
                        self.assertIs(produced.dtype, dtype)
                        self.assertEqual(produced.backend_storage.kind, backend)

                        value.data = ts.Tensor([-5.0], dtype=dtype)
                        produced = computation.forward()
                        self.assertEqual(produced.tolist(), [0.0])
                        self.assertEqual(produced.backend_storage.kind, backend)

                        value.data = ts.Tensor([math.nan], dtype=dtype)
                        self.assertTrue(math.isnan(computation.forward().tolist()[0]))

                        value.data = ts.Tensor([0.0], dtype=dtype)
                        with self.assertRaises(ValueError) as caught:
                            computation.forward()
                        self.assertEqual(
                            str(caught.exception),
                            "relu second derivative is undefined at zero",
                        )
