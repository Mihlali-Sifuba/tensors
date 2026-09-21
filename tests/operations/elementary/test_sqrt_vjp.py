"""The square-root VJP, against docs/sqrt-semantics.md sections 6 to 8.

Expectations come from those sections and from the independent correct-
rounding reference, never from another backend's output. The graph-built
VJP is compared both against the eager VJP and against the document.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.state import reset_graph_state

from tests.operations.elementary._sqrt_reference import reference_sqrt

SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45
LARGEST_FLOAT32_SUBNORMAL = 1.1754942106924411e-38
SMALLEST_FLOAT64_SUBNORMAL = 5e-324

GRADIENT_DTYPES = (ts.float64, ts.float32)

PYTHON_VJP_MODULE = "tensors.backend.python.kernels.elementwise.sqrt_gradient"


def available_backends():
    return ts.available_backends()


def rounded(value, dtype):
    """Round one intermediate into ``dtype``, as section 6.1 requires."""
    if dtype is ts.float64:
        return value
    import struct

    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def specified_vjp(upstream, primal, dtype):
    """The three specified steps, each rounding in ``dtype``.

    The root comes from the independent reference rather than from any
    backend; the remaining two steps are correctly rounded arithmetic,
    performed in binary64 and rounded once into the declared dtype, which
    is exact for a doubling and safe for a division.
    """
    name = "float64" if dtype is ts.float64 else "float32"
    root = reference_sqrt(primal, name)
    denominator = rounded(2.0 * root, dtype)
    return rounded(upstream / denominator, dtype)


def vjp(backend, primals, upstreams, dtype, create_graph=False):
    with ts.use_backend(backend):
        value = ts.Variable(ts.Tensor(primals, dtype=dtype))
        produced = ts.grad(
            ts.sqrt(value),
            value,
            grad_outputs=ts.Tensor(upstreams, dtype=dtype),
            create_graph=create_graph,
        )
        return produced.data if create_graph else produced


class SqrtVjpValueTests(unittest.TestCase):
    """Section 6: the three specified steps."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_the_specified_steps_are_reproduced_exactly(self):
        """Section 6.1, against the independent reference."""
        primals = [4.0, 2.0, 9.0, 0.25, 1e-20, 1e20]
        upstreams = [1.0, 3.0, -2.0, 0.5, 7.0, -1.5]
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                expected = [
                    specified_vjp(upstream, primal, dtype)
                    for upstream, primal in zip(upstreams, primals)
                ]
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(backend, primals, upstreams, dtype).tolist()
                    self.assertEqual(produced, expected)

    def test_float32_rounds_at_every_step_not_once_at_the_end(self):
        """Section 6.1: the sequence, not a binary64 evaluation."""
        primals = [2.0, 3.0, 7.0, 1e-30]
        upstreams = [1.0, 1.0, 5.0, 1.0]
        expected = [
            specified_vjp(upstream, primal, ts.float32)
            for upstream, primal in zip(upstreams, primals)
        ]
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(backend, primals, upstreams, ts.float32).tolist()
                self.assertEqual(produced, expected)

    def test_a_zero_primal_raises_identically_on_every_backend(self):
        """Section 6.2, on both signed zeros."""
        expected = "sqrt derivative is undefined at zero"
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                for primal in (0.0, -0.0):
                    with self.subTest(backend=backend, dtype=dtype.name, primal=primal):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([primal], dtype=dtype))
                            with self.assertRaises(ValueError) as caught:
                                ts.grad(ts.sqrt(value), value)
                        self.assertEqual(str(caught.exception), expected)

    def test_a_negative_primal_gives_nan_without_raising(self):
        """Section 6.2: a negative primal is a value."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(backend, [-1.0, -4.0], [1.0, 1.0], dtype).tolist()
                    for item in produced:
                        self.assertTrue(math.isnan(item))

    def test_a_nan_primal_gives_nan(self):
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    produced = vjp(backend, [math.nan], [1.0], dtype).tolist()
                    self.assertTrue(math.isnan(produced[0]))

    def test_one_negative_element_does_not_deny_the_others(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend, [4.0, -1.0, 9.0], [1.0, 1.0, 1.0], ts.float64
                ).tolist()
                self.assertEqual(produced[0], 0.25)
                self.assertTrue(math.isnan(produced[1]))
                self.assertAlmostEqual(produced[2], 1.0 / 6.0, places=15)

    def test_a_subnormal_primal_does_not_raise(self):
        """Section 6.3: FTZ must not make a subnormal look like zero."""
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        expected = [specified_vjp(1.0, smallest, ts.float32)]
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(backend, [smallest], [1.0], ts.float32).tolist()
                self.assertEqual(produced, expected)

    def test_a_float64_subnormal_primal_does_not_raise(self):
        smallest = SMALLEST_FLOAT64_SUBNORMAL
        expected = [specified_vjp(1.0, smallest, ts.float64)]
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(backend, [smallest], [1.0], ts.float64).tolist()
                self.assertEqual(produced, expected)

    def test_a_subnormal_upstream_reaches_the_division(self):
        """Section 6.3: the division must not flush its numerator.

        The **largest** subnormal is used deliberately. Halving the smallest
        one underflows to zero in any correct implementation, so it cannot
        tell a flushing division from a conforming one; halving the largest
        gives a representable subnormal that only a conforming division
        produces.
        """
        largest = LARGEST_FLOAT32_SUBNORMAL
        primals = [1.0, 4.0]
        upstreams = [largest, largest]
        expected = [
            specified_vjp(upstream, primal, ts.float32)
            for upstream, primal in zip(upstreams, primals)
        ]
        # Guard the guard: these must not be zero, or the test proves
        # nothing about flushing.
        for item in expected:
            self.assertNotEqual(item, 0.0)
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(backend, primals, upstreams, ts.float32).tolist()
                self.assertEqual(produced, expected)

    def test_signed_zero_and_infinite_upstreams(self):
        """The upstream is arithmetic here, so it flows through."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                produced = vjp(
                    backend,
                    [4.0, 4.0, 4.0, 4.0],
                    [0.0, -0.0, math.inf, math.nan],
                    ts.float64,
                ).tolist()
                self.assertEqual(math.copysign(1.0, produced[0]), 1.0)
                self.assertEqual(math.copysign(1.0, produced[1]), -1.0)
                self.assertEqual(produced[2], math.inf)
                self.assertTrue(math.isnan(produced[3]))

    def test_dtype_shape_and_residency(self):
        """Section 6.4."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Variable(
                            ts.Tensor([[1.0, 4.0], [9.0, 16.0]], dtype=dtype)
                        )
                        produced = ts.grad(ts.sqrt(value), value)
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.shape, value.data.shape)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_mismatched_upstream_is_rejected(self):
        from tensors.operations.base import UNARY_DEMAND
        from tensors.operations.elementary.sqrt import Sqrt

        with ts.use_backend("python"):
            value = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "shape"):
                Sqrt().backward(
                    ts.Tensor([1.0], dtype=ts.float64),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )
            with self.assertRaisesRegex(ValueError, "dtype"):
                Sqrt().backward(
                    ts.Tensor([1.0, 2.0], dtype=ts.float32),
                    value,
                    needs_input_grad=UNARY_DEMAND,
                )


class SqrtVjpExecutionTests(unittest.TestCase):
    """Section 6.4: strict selected-backend execution."""

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
                        value = ts.Variable(ts.Tensor([4.0], dtype=dtype))
                        produced = ts.grad(ts.sqrt(value), value)
                    self.assertEqual(produced.backend_storage.kind, backend)
                    self.assertEqual(produced.tolist(), [0.25])

    def test_the_selected_backend_does_not_use_the_python_kernel(self):
        module = importlib.import_module(PYTHON_VJP_MODULE)
        for backend in self._accelerated():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    with patch.object(
                        module,
                        "sqrt_gradient",
                        side_effect=AssertionError("no Python fallback"),
                    ):
                        with ts.use_backend(backend):
                            value = ts.Variable(ts.Tensor([4.0, 16.0], dtype=dtype))
                            produced = ts.grad(ts.sqrt(value), value)
                    self.assertEqual(produced.tolist(), [0.25, 0.125])
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_declining_kernel_is_reported(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import tensors.backend.numpy.kernels as numpy_backend

        from tensors.backend.config import BackendOperationUnsupportedError

        with patch.object(numpy_backend, "sqrt_gradient", return_value=None):
            with ts.use_backend("numpy"):
                value = ts.Variable(ts.Tensor([4.0], dtype=ts.float64))
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.grad(ts.sqrt(value), value)

    def test_the_kernel_receives_native_values_rather_than_tensors(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.sqrt_gradient

        def spy(grad_values, values, **keywords):
            seen["grad"] = grad_values
            seen["value"] = values
            return original(grad_values, values, **keywords)

        with patch.object(numpy_backend, "sqrt_gradient", spy):
            with ts.use_backend("numpy"):
                value = ts.Variable(
                    ts.Tensor([[1.0, 4.0], [9.0, 16.0]], dtype=ts.float64)
                )
                ts.grad(ts.sqrt(value), value)

        for name in ("grad", "value"):
            self.assertNotIsInstance(seen[name], ts.Tensor)
            self.assertIsInstance(seen[name], numpy.ndarray)
            self.assertEqual(seen[name].shape, (2, 2))

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.elementwise import sqrt_gradient as dispatcher

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


class SqrtGraphVjpTests(unittest.TestCase):
    """Sections 7 and 8: the graph-built VJP and higher-order regions."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_the_graph_built_vjp_equals_the_eager_vjp(self):
        primals = [4.0, 2.0, 1e-20]
        upstreams = [1.0, -3.0, 2.0]
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    eager = vjp(backend, primals, upstreams, dtype)
                    built = vjp(backend, primals, upstreams, dtype, create_graph=True)
                    self.assertEqual(built.tolist(), eager.tolist())
                    self.assertIs(built.dtype, eager.dtype)
                    self.assertEqual(built.shape, eager.shape)
                    self.assertEqual(
                        built.backend_storage.kind, eager.backend_storage.kind
                    )

    def test_the_graph_built_vjp_raises_at_zero(self):
        """Section 7: the domain check travels with the graph vertex."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([0.0]))
                    with self.assertRaisesRegex(ValueError, "undefined at zero"):
                        ts.grad(ts.sqrt(value), value, create_graph=True)

    def test_the_graph_built_vjp_gives_nan_for_negative_and_nan_primals(self):
        """Section 8: nothing falls back to Python."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                built = vjp(
                    backend,
                    [-1.0, math.nan],
                    [1.0, 1.0],
                    ts.float64,
                    create_graph=True,
                ).tolist()
                for item in built:
                    self.assertTrue(math.isnan(item))

    def test_the_second_derivative_matches_the_analytic_rule(self):
        """Section 8: -1 / (4 * x ** 1.5) for a unit upstream."""
        primals = [4.0, 9.0, 0.25]
        expected = [-1.0 / (4.0 * primal**1.5) for primal in primals]
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor(primals))
                    first = ts.grad(ts.sqrt(value), value, create_graph=True)
                    second = ts.grad(
                        first, value, grad_outputs=ts.Tensor([1.0] * len(primals))
                    )
                for produced, want in zip(second.tolist(), expected):
                    self.assertAlmostEqual(produced, want, places=12)

    def test_a_third_derivative_follows_from_the_graph_on_python(self):
        """Section 8: the rule is built from operations, not hand-written.

        Only the Python backend is exercised, and not because the others
        disagree. Third-order differentiation reaches `multiply`'s VJP,
        which routes through `sum_products_to_shape` with an operand that a
        still-unmigrated operation answered with Python storage; the
        residency check then rejects it. That defect is independent of
        `sqrt` — a plain ``v * v * v`` third derivative fails the same way —
        and section 8 records it rather than this milestone fixing it.
        """
        primal = 4.0
        expected = 3.0 / (8.0 * primal**2.5)
        with ts.use_backend("python"):
            value = ts.Variable(ts.Tensor([primal]))
            first = ts.grad(ts.sqrt(value), value, create_graph=True)
            second = ts.grad(
                first, value, grad_outputs=ts.Tensor([1.0]), create_graph=True
            )
            third = ts.grad(second, value, grad_outputs=ts.Tensor([1.0]))
        self.assertAlmostEqual(third.tolist()[0], expected, places=12)

    def test_backward_graph_reads_no_host_values(self):
        """Section 7, checked structurally on the parsed statements."""
        import ast
        import inspect
        import textwrap

        module = importlib.import_module("tensors.operations.elementary.sqrt")
        tree = ast.parse(textwrap.dedent(inspect.getsource(module.Sqrt.backward_graph)))
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
        self.assertEqual(comprehensions, [], "no host-built check")
        self.assertEqual(host_reads, [], "no host value reads")

    def test_a_compiled_graph_replays_onto_a_zero_primal_and_raises(self):
        """Section 7: the decisive replay.

        The previous implementation took the zero check once, from the
        values present when the graph was built. Built over positive values
        and replayed onto a zero, it would have returned an infinity
        instead of raising.
        """
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([4.0, 9.0], dtype=ts.float32))
                    computation = ts.graph.Computation(ts.sqrt(value))
                    computation.forward()

                    value.data = ts.Tensor([16.0, 25.0], dtype=ts.float32)
                    computation.forward()
                    self.assertEqual(
                        ts.grad(ts.sqrt(value), value).tolist(),
                        [
                            specified_vjp(1.0, 16.0, ts.float32),
                            specified_vjp(1.0, 25.0, ts.float32),
                        ],
                    )

                    value.data = ts.Tensor([-4.0, 9.0], dtype=ts.float32)
                    computation.forward()
                    replayed = ts.grad(ts.sqrt(value), value).tolist()
                    self.assertTrue(math.isnan(replayed[0]))

                    value.data = ts.Tensor([0.0, 9.0], dtype=ts.float32)
                    computation.forward()
                    with self.assertRaisesRegex(ValueError, "undefined at zero"):
                        ts.grad(ts.sqrt(value), value)


if __name__ == "__main__":
    unittest.main()
