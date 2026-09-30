"""Reduction operations under strict selected-backend execution."""

import ast
import math
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
from tensors.backend import dispatch as backend_dispatch
import tensors.backend.python.kernels as python_backend
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.computation.gradients import sum_gradient_values
from tensors.graph.state import reset_graph_state
from tests._pairwise_oracle import pairwise_sum

BACKENDS = ("python", "numpy", "cuda")


class ReductionBackendExecutionTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    continue
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def test_forward_reductions_stay_on_the_selection_at_threshold_boundaries(self):
        def body(backend):
            for size in (1, 31, 32, 33, 257):
                values = ts.Tensor([1.0 + index % 3 for index in range(size)])
                for operation in (
                    ts.sum,
                    ts.mean,
                    ts.prod,
                    ts.min,
                    ts.max,
                    ts.variance,
                    ts.std,
                    ts.norm,
                    ts.logsumexp,
                    ts.argmin,
                    ts.argmax,
                ):
                    with self.subTest(operation=operation.__name__, size=size):
                        result = operation(values)
                        self.assertEqual(result.backend_storage.kind, backend)
                        expected_shape = () if operation is ts.norm else (1,)
                        self.assertEqual(result.shape, expected_shape)

        self.for_each_backend(body)

    def test_axes_keepdims_shapes_dtypes_and_integer_reductions(self):
        def body(backend):
            floating = ts.Tensor(
                [float(index + 1) for index in range(24)],
                dtype=ts.float32,
                shape=(2, 3, 4),
            )
            kept = ts.sum(floating, axis=(0, -1), keepdims=True)
            dropped = ts.mean(floating, axis=(0, 2))
            integers = ts.Tensor([1, 2, 3, 4], dtype=ts.int32, shape=(2, 2))
            integer_sum = ts.sum(integers, axis=1)
            self.assertEqual(kept.shape, (1, 3, 1))
            self.assertEqual(dropped.shape, (3,))
            self.assertIs(kept.dtype, ts.float32)
            self.assertIs(integer_sum.dtype, ts.int32)
            self.assertEqual(integer_sum.tolist(), [3, 7])
            for result in (kept, dropped, integer_sum):
                self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_first_order_vjps_stay_on_the_selection(self):
        cases = (
            (ts.mean, [1.0, 2.0, 3.0]),
            (ts.prod, [2.0, 0.0, 4.0]),
            (ts.min, [1.0, 1.0, 2.0]),
            (ts.max, [2.0, 2.0, 1.0]),
            (ts.variance, [1.0, 2.0, 3.0]),
            (ts.std, [1.0, 2.0, 3.0]),
            (ts.norm, [3.0, 4.0]),
            (ts.logsumexp, [1.0, 2.0, 3.0]),
        )

        def body(backend):
            for operation, data in cases:
                with self.subTest(operation=operation.__name__):
                    value = ts.Variable(ts.Tensor(data))
                    gradient = ts.grad(operation(value), value)
                    self.assertEqual(gradient.backend_storage.kind, backend)
                    self.assertEqual(gradient.shape, value.shape)

        self.for_each_backend(body)

    def test_empty_and_nonfinite_semantics_are_native(self):
        def body(backend):
            empty = ts.Tensor([])
            self.assertEqual(ts.sum(empty).tolist(), [0.0])
            self.assertEqual(ts.prod(empty).tolist(), [1.0])
            self.assertTrue(math.isnan(ts.mean(empty).tolist()[0]))
            self.assertTrue(math.isnan(ts.variance(empty).tolist()[0]))
            self.assertTrue(math.isnan(ts.std(empty).tolist()[0]))
            self.assertEqual(ts.norm(empty).tolist(), [0.0])
            self.assertTrue(
                math.isnan(ts.sum(ts.Tensor([math.inf, -math.inf])).tolist()[0])
            )
            self.assertEqual(ts.norm(ts.Tensor([math.inf, 1.0])).tolist(), [math.inf])
            with self.assertRaises(ValueError):
                ts.logsumexp(empty)
            with self.assertRaises(ValueError):
                ts.min(empty)
            with self.assertRaises(ValueError):
                ts.argmax(empty)

        self.for_each_backend(body)

    def test_stability_ties_zeros_and_subnormals(self):
        def body(backend):
            # The pairwise tree overflows to +inf and -inf, then NaN.
            difficult = ts.Tensor([1e308, 1e308, -1e308, -1e308])
            self.assertTrue(math.isnan(ts.sum(difficult).item()))
            tiny = ts.Tensor([1.401298464324817e-45] * 2, dtype=ts.float32)
            self.assertEqual(ts.sum(tiny).tolist(), [2.802596928649634e-45])
            tied = ts.Variable(ts.Tensor([1.0, 1.0, 2.0]))
            self.assertEqual(ts.grad(ts.min(tied), tied).tolist(), [0.5, 0.5, 0.0])
            zero = ts.Variable(ts.Tensor([0.0, 3.0, 4.0]))
            self.assertEqual(ts.grad(ts.prod(zero), zero).tolist(), [12.0, 0.0, 0.0])
            large = ts.Variable(ts.Tensor([1e300, 1e300, 1e-300]))
            produced = ts.grad(ts.prod(large), large).tolist()
            self.assertAlmostEqual(produced[0], 1.0)
            self.assertAlmostEqual(produced[1], 1.0)
            self.assertEqual(produced[2], math.inf)

        self.for_each_backend(body)

    def test_sum_obeys_the_documented_numerical_contract(self):
        """Every backend computes the specified pairwise tree.

        The first three cases are where the tree and the exact sum differ, so
        each expectation is the oracle's tree value and is checked to differ
        from the exact sum: lost units reduce to 0 (exact 16), the temporary
        overflow to NaN (exact 0), and the mixed magnitudes to 0 (exact 3).
        """
        difficult = tuple(
            (name, values, pairwise_sum(values))
            for name, values in (
                ("lost units", [1e16, 1.0, -1e16, 1.0] * 8),
                ("temporary overflow", [1e308, 1e308, -1e308, -1e308]),
                ("mixed magnitudes", [1e300, 1.0, -1e300, 2.0]),
            )
        )
        self.assertEqual(difficult[0][2], 0.0)
        self.assertTrue(math.isnan(difficult[1][2]))
        self.assertEqual(difficult[2][2], 0.0)
        for _, values, expected in difficult:
            exact = float(sum((Fraction(value) for value in values), Fraction(0)))
            self.assertNotEqual(exact, expected)
        supported = (
            ("canonical zero", [1.0, -1.0, -0.0], 0.0),
            ("positive infinity", [math.inf, 1.0], math.inf),
            ("negative infinity", [-math.inf, 1.0], -math.inf),
            ("both infinities", [math.inf, -math.inf], math.nan),
            ("NaN", [math.nan, 1.0], math.nan),
            ("empty", [], 0.0),
        )
        for backend in BACKENDS:
            if backend not in ts.available_backends():
                continue
            with ts.use_backend(backend):
                for name, values, expected in difficult:
                    with self.subTest(backend=backend, case=name):
                        result = ts.sum(ts.Tensor(values))
                        self.assertEqual(result.backend_storage.kind, backend)
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(result.item()))
                        else:
                            self.assertEqual(result.item(), expected)
                for name, values, expected in supported:
                    with self.subTest(backend=backend, case=name):
                        result = ts.sum(ts.Tensor(values)).item()
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(result))
                        else:
                            self.assertEqual(result, expected)
                zero = ts.sum(ts.Tensor([1.0, -1.0, -0.0])).item()
                self.assertEqual(math.copysign(1.0, zero), 1.0)
                smallest = math.ulp(0.0)
                self.assertEqual(
                    ts.sum(ts.Tensor([smallest, smallest])).item(), smallest * 2
                )

    def test_integer_sum_preserves_int64_and_reduction_overflow_raises(self):
        def body(backend):
            exact = ts.Tensor([9007199254740993, 1], dtype=ts.int64)
            self.assertEqual(ts.sum(exact).item(), 9007199254740994)
            maximum = 2**63 - 1
            cancellation = ts.Tensor(
                [maximum, maximum, -maximum, -maximum], dtype=ts.int64
            )
            self.assertEqual(ts.sum(cancellation).item(), 0)
            with self.assertRaises(OverflowError):
                ts.sum(ts.Tensor([2**31 - 1, 1], dtype=ts.int32))
            with self.assertRaises(OverflowError):
                ts.prod(ts.Tensor([2**30, 2], dtype=ts.int32))
            with self.assertRaises(OverflowError):
                ts.sum(ts.Tensor([2**63 - 1, 1], dtype=ts.int64))
            with self.assertRaises(OverflowError):
                ts.prod(ts.Tensor([2**62, 2], dtype=ts.int64))
            with self.assertRaises(OverflowError):
                ts.prod(ts.Tensor([2**32, 2**32], dtype=ts.int64))
            self.assertEqual(
                ts.prod(ts.Tensor([2**62, 2, 0], dtype=ts.int64)).item(), 0
            )

        self.for_each_backend(body)

    def test_broadcast_gradient_sums_share_the_public_summation_contract(self):
        """Gradient reductions use the same tree as ``ts.sum``, on every backend.

        The broadcast reduction, the accumulation of separate contributions,
        and the derivative through ``+`` all reduce the same 32 hostile values
        in the same order, so all three give the oracle's tree value, 0; the
        exact sum is 16. Nothing is refused.
        """
        hostile = [1e16, 1.0, -1e16, 1.0] * 8
        expected = pairwise_sum(hostile)
        self.assertEqual(expected, 0.0)
        for backend in BACKENDS:
            if backend not in ts.available_backends():
                continue
            reset_graph_state()
            with ts.use_backend(backend):
                gradient = ts.Tensor(hostile)
                contributions = [ts.Tensor([value]) for value in hostile]
                value = ts.Variable(ts.Tensor([1.0]))
                output = value + ts.Tensor([0.0] * len(hostile))
                storage = backend_dispatch.execute_sum_to_shape(gradient, (1,))
                self.assertEqual(storage.kind, backend)
                self.assertEqual(list(storage.buffer), [expected])
                accumulated = sum_gradient_values(contributions)
                self.assertEqual(accumulated.backend_storage.kind, backend)
                self.assertEqual(accumulated.item(), expected)
                derivative = ts.grad(output, value, grad_outputs=gradient)
                self.assertEqual(derivative.backend_storage.kind, backend)
                self.assertEqual(derivative.item(), expected)

    def test_accelerated_exact_reductions_have_no_python_control_flow_loops(self):
        repository = Path(__file__).resolve().parents[3]
        for backend in ("numpy", "cuda"):
            with self.subTest(backend=backend):
                source = (
                    repository
                    / "tensors"
                    / "backend"
                    / backend
                    / "kernels"
                    / "reductions"
                    / "exact.py"
                ).read_text(encoding="utf-8")
                tree = ast.parse(source)
                loops = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
                ]
                self.assertEqual(loops, [])

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_integer_product_uses_native_cumulative_accumulation(self):
        with (
            patch.object(numpy, "cumprod", wraps=numpy.cumprod) as cumulative,
            patch.object(
                python_backend,
                "reduce_prod",
                side_effect=AssertionError("Python fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            value = ts.Tensor([2**62, 2, 0], dtype=ts.int64)
            self.assertEqual(ts.prod(value).item(), 0)
        cumulative.assert_called_once()

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_dispatch_lowers_to_native_values_and_never_calls_python(self):
        seen = []
        original = numpy_backend.reduce_sum

        def spy(values, *args, **kwargs):
            seen.append(values)
            self.assertNotIsInstance(values, ts.Tensor)
            return original(values, *args, **kwargs)

        with (
            patch.object(numpy_backend, "reduce_sum", side_effect=spy),
            patch.object(
                python_backend,
                "reduce_sum",
                side_effect=AssertionError("Python fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            for size in (1, 31, 32, 33):
                result = ts.sum(ts.Tensor([1.0] * size))
                self.assertEqual(result.backend_storage.kind, "numpy")
        self.assertEqual(len(seen), 4)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_selected_kernel_decline_is_reported(self):
        with (
            patch.object(numpy_backend, "reduce_sum", return_value=None),
            ts.use_backend("numpy"),
        ):
            with self.assertRaises(BackendOperationUnsupportedError):
                ts.sum(ts.Tensor([1.0]))

    def test_small_gradient_accumulation_stays_on_the_selection(self):
        def body(backend):
            contributions = [ts.Tensor([1.0, 2.0]), ts.Tensor([3.0, 4.0])]
            total = sum_gradient_values(contributions)
            self.assertEqual(total.tolist(), [4.0, 6.0])
            self.assertEqual(total.backend_storage.kind, backend)

        self.for_each_backend(body)


if __name__ == "__main__":
    unittest.main()
