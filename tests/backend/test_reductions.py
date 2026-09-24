import unittest
from unittest.mock import patch
import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
from tests.backend._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyReductionTests(NumPyParityTestCase):
    """Reductions, selection, and broadcast gradient reductions."""

    def test_remaining_reductions_and_selection_dispatch_to_numpy(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            names = [
                "reduce_std",
                "reduce_std_gradient",
                "reduce_prod",
                "reduce_prod_gradient",
                "reduce_min",
                "reduce_min_gradient",
                "reduce_max",
                "reduce_max_gradient",
                "argmin",
                "argmax",
                "equal",
                "not_equal",
                "less",
                "less_equal",
                "greater",
                "greater_equal",
                "where",
                "where_gradient",
                "clip",
                "clip_gradient",
                "minimum",
                "maximum",
                "minimum_gradient",
                "maximum_gradient",
            ]
            mocks = {
                name: stack.enter_context(
                    patch.object(
                        numpy_backend, name, wraps=getattr(numpy_backend, name)
                    )
                )
                for name in names
            }
            with ts.use_backend("numpy"):
                value_data = ts.Tensor(
                    [1.0 + index % 8 / 10.0 for index in range(64)], shape=(8, 8)
                )
                for operation in (ts.std, ts.prod, ts.min, ts.max):
                    value = ts.Variable(value_data)
                    output = operation(value, axis=1)
                    ts.grad(output, value, grad_outputs=ts.full((8,), 1.0))
                ts.argmin(value_data, axis=1)
                ts.argmax(value_data, axis=1)
                right = ts.full((8, 8), 1.4)
                for operation in (
                    ts.equal,
                    ts.not_equal,
                    ts.less,
                    ts.less_equal,
                    ts.greater,
                    ts.greater_equal,
                ):
                    operation(value_data, right)
                condition = ts.Tensor(
                    [index % 2 for index in range(64)], dtype=ts.uint8, shape=(8, 8)
                )
                selected = ts.Variable(value_data)
                chosen = ts.where(condition, selected, right)
                ts.grad(chosen, selected, grad_outputs=ts.full((8, 8), 1.0))
                clipped = ts.Variable(value_data)
                clipped_output = ts.clip(clipped, 1.2, 1.6)
                ts.grad(clipped_output, clipped, grad_outputs=ts.full((8, 8), 1.0))
                for operation in (ts.minimum, ts.maximum):
                    selected = ts.Variable(value_data)
                    output = operation(selected, right)
                    ts.grad(output, selected, grad_outputs=ts.full((8, 8), 1.0))
        for name, kernel in mocks.items():
            with self.subTest(kernel=name):
                self.assertGreaterEqual(kernel.call_count, 1)

    def test_remaining_reductions_and_selection_match_python_backend(self):

        def evaluate(backend):
            with ts.use_backend(backend):
                data = ts.Tensor(
                    [1.0 + index % 8 / 10.0 for index in range(64)], shape=(8, 8)
                )
                results = []
                for operation in (ts.std, ts.prod, ts.min, ts.max):
                    value = ts.Variable(data)
                    output = operation(value, axis=1)
                    gradient = ts.grad(output, value, grad_outputs=ts.full((8,), 1.0))
                    results.extend((output.data, gradient))
                results.extend(
                    (
                        ts.argmin(data, axis=1),
                        ts.argmax(data, axis=1),
                        ts.greater_equal(data, 1.4),
                        ts.where(ts.greater(data, 1.4), data, 1.4),
                        ts.clip(data, 1.2, 1.6),
                        ts.minimum(data, 1.4),
                        ts.maximum(data, 1.4),
                    )
                )
                return results

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data, expected_tensor._data
            ):
                self.assertAlmostEqual(actual_item, expected_item)

    def test_reductions_dispatch_to_numpy(self):
        for name in ("sum", "mean", "variance", "norm"):
            with self.subTest(operation=name):
                kernel_name = "reduce_" + name
                with (
                    patch.object(
                        numpy_backend,
                        kernel_name,
                        wraps=getattr(numpy_backend, kernel_name),
                    ) as kernel,
                    ts.use_backend("numpy"),
                ):
                    value = ts.Tensor([float(index + 1) for index in range(512)])
                    getattr(ts, name)(value)
                kernel.assert_called_once()

    def test_axis_reductions_match_python_backend(self):
        def evaluate(backend, operation):
            with ts.use_backend(backend):
                value = ts.Tensor(
                    [float(index + 1) for index in range(24)], shape=(2, 3, 4)
                )
                return operation(value, axis=(0, 2), keepdims=True).tolist()

        for operation in (ts.sum, ts.mean, ts.variance, ts.norm):
            with self.subTest(operation=operation.__name__):
                self.assertEqual(
                    evaluate("numpy", operation), evaluate("python", operation)
                )

    def test_stable_reductions_execute_natively_without_changing_results(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                value = ts.Tensor([1e308, 1e308, -1e308, -1e308])
                smallest = ts.Tensor([5e-324, 5e-324])
                return (
                    ts.sum(value).tolist(),
                    ts.mean(smallest).tolist(),
                    ts.variance(value).tolist(),
                    ts.norm(value).tolist(),
                )

        self.assertEqual(evaluate("numpy"), evaluate("python"))

    def test_broadcast_gradient_reductions_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend, "reduce_sum", wraps=numpy_backend.reduce_sum
            ) as sum_kernel,
            patch.object(
                numpy_backend,
                "sum_products_to_shape",
                wraps=numpy_backend.sum_products_to_shape,
            ) as product_kernel,
            ts.use_backend("numpy"),
        ):
            left = ts.Variable(ts.full((64, 1), 2.0))
            right = ts.Variable(ts.full((1, 64), 3.0))
            ts.grad(ts.sum(left + right), (left, right))
            ts.grad(ts.sum(left * right), (left, right))
        self.assertGreaterEqual(sum_kernel.call_count, 3)
        self.assertGreaterEqual(product_kernel.call_count, 2)

    def test_broadcast_gradient_reductions_match_python_backend(self):

        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.full((64, 1), 2.0))
                right = ts.Variable(ts.full((1, 64), 3.0))
                added = ts.grad(ts.sum(left + right), (left, right))
                multiplied = ts.grad(ts.sum(left * right), (left, right))
                return (
                    tuple((item.tolist() for item in added)),
                    tuple((item.tolist() for item in multiplied)),
                )

        self.assertEqual(gradients("numpy"), gradients("python"))

    def test_product_reduction_preserves_exact_cancellation(self):

        def gradient(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.Tensor([[1.0], [1.0]]))
                right = ts.Variable(ts.Tensor([[1e308, -1e308]]))
                return ts.grad(ts.sum(left * right), left)

        self.assertEqual(gradient("numpy").tolist(), gradient("python").tolist())


if __name__ == "__main__":
    unittest.main()
