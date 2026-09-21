"""The sign VJP, against docs/sign-semantics.md sections 6 to 8.

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
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

#: Only these may require gradients; no integer differentiation is defined.
GRADIENT_DTYPES = (ts.float64, ts.float32)

#: Section 6.1: the upstreams that used to change the answer.
HOSTILE_UPSTREAMS = (-2.0, 2.0, 0.0, -0.0, math.inf, -math.inf, math.nan)

PYTHON_VJP_MODULE = "tensors.backend.python.kernels.elementwise.sign_gradient"


def available_backends():
    return ts.available_backends()


def is_positive_zero(value):
    return value == 0.0 and math.copysign(1.0, value) == 1.0


class SignVjpValueTests(unittest.TestCase):
    """Section 6: the routed first-order VJP."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_finite_nonzero_gives_canonical_positive_zero(self):
        """Section 6: the result is +0.0 at every finite nonzero primal."""
        primals = [2.0, -3.5, 1e-30, -1e30]
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(ts.Tensor(primals, dtype=dtype))
                        produced = ts.grad(ts.sign(value), value).tolist()
                    for item in produced:
                        self.assertTrue(is_positive_zero(item))

    def test_the_upstream_gradient_cannot_change_the_result(self):
        """Section 6.1: routing, not multiplication."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                for upstream in HOSTILE_UPSTREAMS:
                    with self.subTest(
                        backend=backend, dtype=dtype.name, upstream=upstream
                    ):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([2.0, -2.0], dtype=dtype))
                            produced = ts.grad(
                                ts.sign(value),
                                value,
                                grad_outputs=ts.Tensor(
                                    [upstream, upstream], dtype=dtype
                                ),
                            ).tolist()
                        for item in produced:
                            self.assertTrue(
                                is_positive_zero(item),
                                f"upstream {upstream!r} leaked into the result",
                            )

    def test_a_subnormal_upstream_gradient_is_also_routed_away(self):
        """Section 6.1, with an upstream small enough to be flushed."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([2.0, -2.0], dtype=ts.float32))
                    produced = ts.grad(
                        ts.sign(value),
                        value,
                        grad_outputs=ts.Tensor([smallest, -smallest], dtype=ts.float32),
                    ).tolist()
                for item in produced:
                    self.assertTrue(is_positive_zero(item))

    def test_a_nan_primal_gives_nan(self):
        """Section 6: classification only."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(ts.Tensor([math.nan], dtype=dtype))
                        produced = ts.grad(ts.sign(value), value).tolist()
                    self.assertTrue(math.isnan(produced[0]))

    def test_a_subnormal_primal_is_nonzero_and_does_not_raise(self):
        """Section 6.3: FTZ must not make a subnormal look like zero."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(
                        ts.Tensor(
                            [SMALLEST_FLOAT32_SUBNORMAL, -SMALLEST_FLOAT32_SUBNORMAL],
                            dtype=ts.float32,
                        )
                    )
                    produced = ts.grad(ts.sign(value), value).tolist()
                for item in produced:
                    self.assertTrue(is_positive_zero(item))

    def test_a_float64_subnormal_primal_is_nonzero(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(
                        ts.Tensor([SMALLEST_FLOAT64_SUBNORMAL], dtype=ts.float64)
                    )
                    produced = ts.grad(ts.sign(value), value).tolist()
                self.assertTrue(is_positive_zero(produced[0]))

    def test_a_zero_primal_raises_identically_on_every_backend(self):
        """Section 6: the exact error, on both signed zeros."""
        expected = "sign derivative is undefined at zero"
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                for primal in (0.0, -0.0):
                    with self.subTest(backend=backend, dtype=dtype.name, primal=primal):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([primal], dtype=dtype))
                            with self.assertRaises(ValueError) as caught:
                                ts.grad(ts.sign(value), value)
                        self.assertEqual(str(caught.exception), expected)

    def test_dtype_shape_and_residency(self):
        """Section 6.2."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(
                            ts.Tensor([[1.0, -2.0], [3.0, -4.0]], dtype=dtype)
                        )
                        produced = ts.grad(ts.sign(value), value)
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.shape, value.data.shape)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_mismatched_upstream_is_rejected(self):
        """Section 6.2: shape and dtype agreement is Tensor semantics."""
        from tensors.operations.elementary.sign import Sign
        from tensors.operations.base import UNARY_DEMAND

        with ts.use_backend("python"):
            value = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "shape"):
                Sign().backward(
                    ts.Tensor([1.0], dtype=ts.float64),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )
            with self.assertRaisesRegex(ValueError, "dtype"):
                Sign().backward(
                    ts.Tensor([1.0, 2.0], dtype=ts.float32),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )


class SignVjpExecutionTests(unittest.TestCase):
    """Section 6.2: strict selected-backend execution."""

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
                        produced = ts.grad(ts.sign(value), value)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_kernel(self):
        module = importlib.import_module(PYTHON_VJP_MODULE)
        for backend in self._accelerated():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "sign_gradient",
                        side_effect=AssertionError("no Python fallback"),
                    ):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([3.0, -1.0], dtype=dtype))
                            produced = ts.grad(ts.sign(value), value)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_declining_kernel_is_reported(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import tensors.backend.numpy.kernels as numpy_backend

        from tensors.backend.config import BackendOperationUnsupportedError

        with patch.object(numpy_backend, "sign_gradient", return_value=None):
            with ts.use_backend("numpy"):
                value = ts.Variable(ts.Tensor([1.0], dtype=ts.float64))
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.grad(ts.sign(value), value)

    def test_the_kernel_receives_native_values_rather_than_tensors(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.sign_gradient

        def spy(grad_values, values, **keywords):
            seen["grad"] = grad_values
            seen["value"] = values
            return original(grad_values, values, **keywords)

        with patch.object(numpy_backend, "sign_gradient", spy):
            with ts.use_backend("numpy"):
                value = ts.Variable(
                    ts.Tensor([[1.0, -2.0], [3.0, -4.0]], dtype=ts.float64)
                )
                ts.grad(ts.sign(value), value)

        for name in ("grad", "value"):
            self.assertNotIsInstance(seen[name], ts.Tensor)
            self.assertIsInstance(seen[name], numpy.ndarray)
            self.assertEqual(seen[name].shape, (2, 2))

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.elementwise import sign_gradient as dispatcher

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


class SignGraphVjpTests(unittest.TestCase):
    """Sections 7 and 8: the graph-built VJP and higher-order regions."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_the_graph_built_vjp_equals_the_eager_vjp(self):
        """Section 7, including dtype, shape and residency."""
        primals = [2.0, -3.0, 1e-20]
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(ts.Tensor(primals, dtype=dtype))
                        eager = ts.grad(ts.sign(value), value)
                        built = ts.grad(ts.sign(value), value, create_graph=True).data
                    self.assertEqual(built.tolist(), eager.tolist())
                    for item in built.tolist():
                        self.assertTrue(is_positive_zero(item))
                    self.assertIs(built.dtype, eager.dtype)
                    self.assertEqual(built.shape, eager.shape)
                    self.assertEqual(
                        built.backend_storage.kind, eager.backend_storage.kind
                    )

    def test_the_graph_built_vjp_propagates_a_nan_primal(self):
        """Section 8: NaN is not silently turned into zero."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([math.nan]))
                    built = ts.grad(ts.sign(value), value, create_graph=True).data
                self.assertTrue(math.isnan(built.tolist()[0]))

    def test_the_graph_built_vjp_raises_at_zero(self):
        """Section 7: the domain error survives graph construction."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([0.0]))
                    with self.assertRaisesRegex(ValueError, "undefined at zero"):
                        ts.grad(ts.sign(value), value, create_graph=True)

    def test_higher_derivatives_are_zero_away_from_zero(self):
        """Section 8: the second and third derivatives."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([-2.0, 3.0]))
                    first = ts.grad(ts.sign(value), value, create_graph=True)
                    second = ts.grad(
                        first,
                        value,
                        grad_outputs=ts.Tensor([1.0, 1.0]),
                        create_graph=True,
                    )
                    third = ts.grad(second, value, grad_outputs=ts.Tensor([1.0, 1.0]))
                self.assertEqual(first.data.tolist(), [0.0, 0.0])
                self.assertEqual(second.data.tolist(), [0.0, 0.0])
                self.assertEqual(third.tolist(), [0.0, 0.0])

    def test_backward_graph_reads_no_host_values(self):
        """Section 7: no Python mask is built from the materialised data.

        Checked structurally on the parsed statements rather than by
        substring: the docstring deliberately names ``value.data._data`` to
        record what the previous implementation did.
        """
        import ast
        import inspect
        import textwrap

        module = importlib.import_module("tensors.operations.elementary.sign")
        tree = ast.parse(textwrap.dedent(inspect.getsource(module.Sign.backward_graph)))
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

    def test_a_compiled_graph_replays_with_values_that_change_branch(self):
        """Section 7: a frozen mask would answer for the build-time values."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([2.0, 3.0], dtype=ts.float32))
                    computation = ts.graph.Computation(ts.sign(value))
                    computation.forward()

                    for replacement in ([-2.0, -3.0], [smallest, -smallest]):
                        value.data = ts.Tensor(replacement, dtype=ts.float32)
                        computation.forward()
                        produced = ts.grad(ts.sign(value), value).tolist()
                        for item in produced:
                            self.assertTrue(is_positive_zero(item))

                    # Replaying onto a zero must raise, which a frozen mask
                    # built over positive values could not know to do.
                    value.data = ts.Tensor([0.0, 1.0], dtype=ts.float32)
                    computation.forward()
                    with self.assertRaisesRegex(ValueError, "undefined at zero"):
                        ts.grad(ts.sign(value), value)


if __name__ == "__main__":
    unittest.main()
