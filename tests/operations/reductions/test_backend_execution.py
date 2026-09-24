"""Reduction operations under strict selected-backend execution."""

import math
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.computation.gradients import sum_gradient_values
from tensors.graph.state import reset_graph_state

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
            self.assertEqual(
                ts.sum(ts.Tensor([1e308, 1e308, -1e308, -1e308])).tolist(),
                [0.0],
            )
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
