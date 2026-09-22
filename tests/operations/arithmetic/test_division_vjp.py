"""The division VJP under the selected-backend execution contract.

`docs/arithmetic-semantics.md` section 7.4. Expectations are derived from
that section and from the division rules of section 7.2, never by running
one backend and recording what it gave.
"""

import importlib
import math
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.state import reset_graph_state

SMALLEST_FLOAT32_SUBNORMAL = 1.401298464324817e-45

GRADIENT_DTYPES = (ts.float64, ts.float32)

PYTHON_VJP_MODULE = (
    "tensors.backend.python.kernels.elementwise.division_denominator_gradient"
)


def available_backends():
    return ts.available_backends()


def divide_gradients(backend, numerator, denominator, upstream, dtype, **keywords):
    """Both VJPs of ``numerator / denominator`` at one upstream gradient."""
    with ts.use_backend(backend):
        a = ts.Variable(ts.Tensor(numerator, dtype=dtype))
        b = ts.Variable(ts.Tensor(denominator, dtype=dtype))
        return ts.grad(
            a / b,
            [a, b],
            grad_outputs=ts.Tensor(upstream, dtype=dtype),
            **keywords,
        )


class DivisionVjpResidencyTests(unittest.TestCase):
    """Section 7.4: the VJP executes where the selection says."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def _accelerated(self):
        return [b for b in ts.available_backends() if b in ("numpy", "cuda")]

    def test_a_one_element_backward_stays_on_the_selected_backend(self):
        """Below the old NumPy threshold, which used to answer in Python."""
        for backend in self._accelerated():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    numerator, denominator = divide_gradients(
                        backend, [6.0], [2.0], [1.0], dtype
                    )
                    self.assertEqual(numerator.tolist(), [0.5])
                    self.assertEqual(denominator.tolist(), [-1.5])
                    self.assertEqual(numerator.backend_storage.kind, backend)
                    self.assertEqual(denominator.backend_storage.kind, backend)

    def test_a_broadcast_backward_stays_on_the_selected_backend(self):
        """Both reductions, which the legacy sum_to_shape answered in Python."""
        for backend in self._accelerated():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([[6.0, 8.0]], dtype=ts.float64))
                    b = ts.Variable(ts.Tensor([[2.0], [4.0]], dtype=ts.float64))
                    numerator, denominator = ts.grad(
                        a / b,
                        [a, b],
                        grad_outputs=ts.Tensor(
                            [[1.0, 1.0], [1.0, 1.0]], dtype=ts.float64
                        ),
                    )
                self.assertEqual(numerator.shape, a.data.shape)
                self.assertEqual(denominator.shape, b.data.shape)
                self.assertEqual(numerator.backend_storage.kind, backend)
                self.assertEqual(denominator.backend_storage.kind, backend)

    def test_only_the_numerator_or_only_the_denominator(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([6.0]), requires_grad=True)
                    fixed_b = ts.Variable(ts.Tensor([2.0]), requires_grad=False)
                    only_numerator = ts.grad(
                        a / fixed_b, a, grad_outputs=ts.Tensor([1.0])
                    )

                    fixed_a = ts.Variable(ts.Tensor([6.0]), requires_grad=False)
                    b = ts.Variable(ts.Tensor([2.0]), requires_grad=True)
                    only_denominator = ts.grad(
                        fixed_a / b, b, grad_outputs=ts.Tensor([1.0])
                    )
                self.assertEqual(only_numerator.tolist(), [0.5])
                self.assertEqual(only_denominator.tolist(), [-1.5])
                self.assertEqual(only_numerator.backend_storage.kind, backend)
                self.assertEqual(only_denominator.backend_storage.kind, backend)

    def test_the_selected_backend_does_not_use_the_python_kernel(self):
        module = importlib.import_module(PYTHON_VJP_MODULE)
        for backend in self._accelerated():
            with self.subTest(backend=backend):
                with patch.object(
                    module,
                    "division_denominator_gradient",
                    side_effect=AssertionError("no Python fallback"),
                ):
                    numerator, denominator = divide_gradients(
                        backend, [6.0], [2.0], [1.0], ts.float64
                    )
                self.assertEqual(denominator.tolist(), [-1.5])
                self.assertEqual(denominator.backend_storage.kind, backend)

    def test_a_declining_kernel_is_reported(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import tensors.backend.numpy.kernels as numpy_backend

        from tensors.backend.config import BackendOperationUnsupportedError

        with patch.object(
            numpy_backend, "division_denominator_gradient", return_value=None
        ):
            with ts.use_backend("numpy"):
                a = ts.Variable(ts.Tensor([6.0]))
                b = ts.Variable(ts.Tensor([2.0]))
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.grad(a / b, [a, b], grad_outputs=ts.Tensor([1.0]))

    def test_the_kernel_receives_native_values_rather_than_tensors(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy

        import tensors.backend.numpy.kernels as numpy_backend

        seen = {}
        original = numpy_backend.division_denominator_gradient

        def spy(grad_values, numerator_values, denominator_values, **keywords):
            seen["operands"] = (grad_values, numerator_values, denominator_values)
            return original(
                grad_values, numerator_values, denominator_values, **keywords
            )

        with patch.object(numpy_backend, "division_denominator_gradient", spy):
            with ts.use_backend("numpy"):
                a = ts.Variable(ts.Tensor([[6.0, 8.0]], dtype=ts.float64))
                b = ts.Variable(ts.Tensor([[2.0, 4.0]], dtype=ts.float64))
                ts.grad(a / b, [a, b], grad_outputs=ts.Tensor([[1.0, 1.0]]))

        for operand in seen["operands"]:
            self.assertNotIsInstance(operand, ts.Tensor)
            self.assertIsInstance(operand, numpy.ndarray)
            self.assertEqual(operand.shape, (1, 2))


class DivisionVjpBoundaryTests(unittest.TestCase):
    """The removed legacy machinery must not reappear."""

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.elementwise import (
            division_denominator_gradient as dispatcher,
        )

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
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)

    def test_the_array_kernels_do_not_lower_their_own_operands(self):
        import inspect

        for module_name in (
            "tensors.backend.numpy.kernels.elementwise.division_denominator_gradient",
            "tensors.backend.cuda.kernels.elementwise.division_denominator_gradient",
        ):
            with self.subTest(module=module_name):
                source = inspect.getsource(importlib.import_module(module_name))
                self.assertNotIn("tensor_to_logical_array", source)
                self.assertNotIn("._data", source)

    def test_the_division_backward_does_not_use_the_legacy_reduction(self):
        """The broadcast reduction must be the selected-backend one."""
        import inspect

        module = importlib.import_module("tensors.operations.arithmetic.divide")
        source = inspect.getsource(module)
        self.assertIn("sum_to_shape_graph_on_selected_backend", source)
        # Neither reduction that consults the workload policy may appear.
        remainder = source.replace("sum_to_shape_graph_on_selected_backend(", "")
        self.assertNotIn("sum_to_shape(", remainder)
        self.assertNotIn("sum_to_shape_graph(", remainder)

    def test_the_denominator_vjp_does_not_scan_host_values(self):
        import ast
        import inspect
        import textwrap

        module = importlib.import_module("tensors.operations.arithmetic.divide")
        for method in (
            module.DivisionDenominatorGradient.forward,
            module.DivisionDenominatorGradient.backward,
        ):
            with self.subTest(method=method.__name__):
                tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
                host_reads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute) and node.attr == "_data"
                ]
                self.assertEqual(
                    host_reads, [], f"{method.__name__} must not read host values"
                )


class DivisionVjpSemanticsTests(unittest.TestCase):
    """Sections 7.2 and 7.4: the specified values."""

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def test_ordinary_first_order_values(self):
        """g / b and -g * a / b**2."""
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    numerator, denominator = divide_gradients(
                        backend, [6.0, -8.0], [2.0, 4.0], [1.0, 3.0], dtype
                    )
                    self.assertEqual(numerator.tolist(), [0.5, 0.75])
                    self.assertEqual(denominator.tolist(), [-1.5, 1.5])

    def test_a_zero_denominator_follows_section_7_2(self):
        """Signed infinities and NaN, not an exception."""
        expectations = (
            # numerator, denominator, upstream, numerator VJP, denominator VJP
            (1.0, 0.0, 1.0, math.inf, -math.inf),
            (-1.0, 0.0, 1.0, math.inf, math.inf),
            (0.0, 0.0, 1.0, math.inf, None),
        )
        for backend in available_backends():
            with self.subTest(backend=backend):
                for a, b, g, want_a, want_b in expectations:
                    numerator, denominator = divide_gradients(
                        backend, [a], [b], [g], ts.float64
                    )
                    self.assertEqual(numerator.tolist(), [want_a])
                    if want_b is None:
                        self.assertTrue(math.isnan(denominator.tolist()[0]))
                    else:
                        self.assertEqual(denominator.tolist(), [want_b])

    def test_a_negative_zero_denominator(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                numerator, denominator = divide_gradients(
                    backend, [1.0], [-0.0], [1.0], ts.float64
                )
                self.assertEqual(numerator.tolist(), [-math.inf])
                # -g * a / b**2: the square of a signed zero is +0.0, so the
                # sign comes from the numerator alone.
                self.assertEqual(denominator.tolist(), [-math.inf])

    def test_infinite_and_nan_upstream_gradients(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                numerator, denominator = divide_gradients(
                    backend, [6.0, 6.0], [2.0, 2.0], [math.inf, math.nan], ts.float64
                )
                self.assertEqual(numerator.tolist()[0], math.inf)
                self.assertTrue(math.isnan(numerator.tolist()[1]))
                self.assertEqual(denominator.tolist()[0], -math.inf)
                self.assertTrue(math.isnan(denominator.tolist()[1]))

    def test_the_range_safe_denominator_vjp_is_preserved(self):
        """``b ** 2`` overflows here, but the quotient is representable."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                _, denominator = divide_gradients(
                    backend, [1e300], [1e200], [1.0], ts.float64
                )
                produced = denominator.tolist()[0]
                self.assertNotEqual(
                    produced, 0.0, "a naive b*b overflows and gives zero"
                )
                self.assertTrue(math.isfinite(produced))
                self.assertAlmostEqual(produced / -1e-100, 1.0, places=9)

    def test_a_float32_subnormal_upstream_on_every_backend(self):
        smallest = SMALLEST_FLOAT32_SUBNORMAL
        for backend in available_backends():
            with self.subTest(backend=backend):
                numerator, _ = divide_gradients(
                    backend, [4.0], [1.0], [smallest], ts.float32
                )
                self.assertEqual(numerator.tolist(), [smallest])

    def test_eager_and_graph_built_first_vjps_agree(self):
        for backend in available_backends():
            for dtype in GRADIENT_DTYPES:
                with self.subTest(backend=backend, dtype=dtype.name):
                    eager = divide_gradients(
                        backend, [6.0, -8.0], [2.0, 4.0], [1.0, 3.0], dtype
                    )
                    built = divide_gradients(
                        backend,
                        [6.0, -8.0],
                        [2.0, 4.0],
                        [1.0, 3.0],
                        dtype,
                        create_graph=True,
                    )
                    for produced, expected in zip(built, eager):
                        self.assertEqual(produced.data.tolist(), expected.tolist())
                        self.assertEqual(
                            produced.data.backend_storage.kind,
                            expected.backend_storage.kind,
                        )

    def test_eager_and_graph_agree_at_a_zero_denominator(self):
        """The host scan used to make the graph path raise here."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                eager = divide_gradients(backend, [1.0], [0.0], [1.0], ts.float64)
                built = divide_gradients(
                    backend, [1.0], [0.0], [1.0], ts.float64, create_graph=True
                )
                self.assertEqual(
                    built[1].data.tolist(), eager[1].tolist(), "denominator VJP"
                )
                self.assertEqual(eager[1].tolist(), [-math.inf])

    def test_the_second_derivative_matches_the_analytic_rule(self):
        """d2(a/b)/db2 = 2a / b**3, on the selected backend."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([6.0]))
                    b = ts.Variable(ts.Tensor([2.0]))
                    _, first = ts.grad(
                        a / b,
                        [a, b],
                        grad_outputs=ts.Tensor([1.0]),
                        create_graph=True,
                    )
                    second = ts.grad(first, b, grad_outputs=ts.Tensor([1.0]))
                self.assertAlmostEqual(
                    second.tolist()[0], 2.0 * 6.0 / 2.0**3, places=12
                )
                self.assertEqual(second.backend_storage.kind, backend)

    def test_a_compiled_first_vjp_graph_replays_onto_a_zero_denominator(self):
        """The recorded VJP is re-executed, domain behaviour included."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([6.0], dtype=ts.float64))
                    b = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
                    _, denominator = ts.grad(
                        a / b,
                        [a, b],
                        grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                        create_graph=True,
                    )
                    computation = ts.graph.Computation(denominator)
                    self.assertEqual(computation.forward().tolist(), [-1.5])

                    b.data = ts.Tensor([4.0], dtype=ts.float64)
                    produced = computation.forward()
                    self.assertEqual(produced.tolist(), [-0.375])
                    self.assertEqual(produced.backend_storage.kind, backend)

                    b.data = ts.Tensor([0.0], dtype=ts.float64)
                    produced = computation.forward()
                self.assertEqual(produced.tolist(), [-math.inf])
                self.assertEqual(produced.backend_storage.kind, backend)


class DivisionSingleBackwardTests(unittest.TestCase):
    """Division defines one derivative, and it serves both reverse modes.

    Section 7.2 fixes the values, so the expectations here are the quotient
    rule — ``g / b`` and ``-g * a / b**2`` — with that section's conventions
    at a zero denominator. No backend is read to judge another.
    """

    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def test_both_reverse_modes_give_the_same_gradients(self):
        cases = {
            "ordinary": ([6.0, -8.0], [2.0, 4.0], [1.0, 3.0],
                         [0.5, 0.75], [-1.5, 1.5], (2,), (2,)),
            "singleton": ([[6.0, 8.0]], [[2.0], [4.0]], [[1.0, 1.0], [1.0, 1.0]],
                          [0.75, 0.75], [-3.5, -0.875], (1, 2), (2, 1)),
            "leading": ([[6.0, 8.0], [10.0, 12.0]], [2.0, 4.0],
                        [[1.0, 1.0], [1.0, 1.0]],
                        [0.5, 0.25, 0.5, 0.25], [-4.0, -1.25], (2, 2), (2,)),
        }
        for backend in self.BACKENDS:
            for name, (n, d, g, wn, wd, sn, sd) in cases.items():
                for create_graph in (False, True):
                    with self.subTest(backend=backend, case=name,
                                      create_graph=create_graph):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            a = ts.Variable(ts.Tensor(n, dtype=ts.float64))
                            b = ts.Variable(ts.Tensor(d, dtype=ts.float64))
                            gn, gd = ts.grad(
                                a / b, [a, b],
                                grad_outputs=ts.Tensor(g, dtype=ts.float64),
                                create_graph=create_graph,
                            )
                            if create_graph:
                                gn, gd = gn.data, gd.data
                        for produced, want, shape in ((gn, wn, sn), (gd, wd, sd)):
                            self.assertEqual(produced.tolist(), want)
                            self.assertEqual(tuple(produced.shape), shape)
                            self.assertIs(produced.dtype, ts.float64)
                            self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_zero_denominator_keeps_its_section_7_2_values_in_both_modes(self):
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        a = ts.Variable(ts.Tensor([1.0], dtype=ts.float64))
                        b = ts.Variable(ts.Tensor([0.0], dtype=ts.float64))
                        gn, gd = ts.grad(
                            a / b, [a, b],
                            grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                            create_graph=create_graph,
                        )
                        if create_graph:
                            gn, gd = gn.data, gd.data
                    self.assertEqual(gn.tolist(), [math.inf])
                    self.assertEqual(gd.tolist(), [-math.inf])
                    self.assertEqual(gn.backend_storage.kind, backend)

    def test_the_range_safe_denominator_vjp_survives_both_modes(self):
        """``b ** 2`` overflows here; the quotient is representable."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        a = ts.Variable(ts.Tensor([1.0e300], dtype=ts.float64))
                        b = ts.Variable(ts.Tensor([1.0e200], dtype=ts.float64))
                        _, gd = ts.grad(
                            a / b, [a, b],
                            grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                            create_graph=create_graph,
                        )
                        if create_graph:
                            gd = gd.data
                    produced = gd.tolist()[0]
                    self.assertTrue(math.isfinite(produced))
                    self.assertAlmostEqual(produced / -1.0e-100, 1.0, places=9)

    def test_the_second_derivative_and_replay(self):
        """d2(a/b)/db2 is 2a/b**3, and the recorded first VJP re-runs."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([6.0], dtype=ts.float64))
                    b = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
                    _, first = ts.grad(
                        a / b, [a, b],
                        grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                        create_graph=True,
                    )
                    self.assertEqual(first.data.tolist(), [-1.5])
                    second = ts.grad(
                        first, b, grad_outputs=ts.Tensor([1.0], dtype=ts.float64)
                    )
                    self.assertAlmostEqual(
                        second.tolist()[0], 2.0 * 6.0 / 2.0**3, places=12
                    )

                    program = ts.graph.Computation(first)
                    self.assertEqual(program.forward().tolist(), [-1.5])
                    b.data = ts.Tensor([4.0], dtype=ts.float64)
                    replayed = program.forward()
                self.assertEqual(replayed.tolist(), [-0.375])
                self.assertEqual(replayed.backend_storage.kind, backend)

    def test_division_defines_exactly_one_derivative(self):
        from tensors.operations.arithmetic.divide import DivisionDenominatorGradient
        from tensors.ops import Div, Operation

        for operation in (Div, DivisionDenominatorGradient):
            with self.subTest(operation=operation.name):
                self.assertIs(operation.backward_graph, Operation.backward_graph)
                self.assertNotIn("backward_graph", vars(operation))


if __name__ == "__main__":
    unittest.main()
