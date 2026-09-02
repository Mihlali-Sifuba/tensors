import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend

from ._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyReductionTests(NumPyParityTestCase):
    """Reductions, selection, and broadcast gradient reductions."""

    def test_remaining_reductions_and_selection_dispatch_to_numpy(self):
        value_data = ts.Tensor(
            [1.0 + (index % 8) / 10.0 for index in range(64)],
            shape=(8, 8),
        )
        with (
            patch.object(
                numpy_backend,
                "reduction",
                wraps=numpy_backend.reduction,
            ) as reduction,
            patch.object(
                numpy_backend,
                "reduction_gradient",
                wraps=numpy_backend.reduction_gradient,
            ) as reduction_gradient,
            patch.object(
                numpy_backend,
                "arg_extremum",
                wraps=numpy_backend.arg_extremum,
            ) as arg_extremum,
            patch.object(
                numpy_backend,
                "comparison",
                wraps=numpy_backend.comparison,
            ) as comparison,
            patch.object(
                numpy_backend,
                "where",
                wraps=numpy_backend.where,
            ) as where,
            patch.object(
                numpy_backend,
                "where_gradient",
                wraps=numpy_backend.where_gradient,
            ) as where_gradient,
            patch.object(
                numpy_backend,
                "clip",
                wraps=numpy_backend.clip,
            ) as clip,
            patch.object(
                numpy_backend,
                "clip_gradient",
                wraps=numpy_backend.clip_gradient,
            ) as clip_gradient,
            patch.object(
                numpy_backend,
                "extremum",
                wraps=numpy_backend.extremum,
            ) as extremum,
            patch.object(
                numpy_backend,
                "extremum_gradient",
                wraps=numpy_backend.extremum_gradient,
            ) as extremum_gradient,
        ):
            with ts.use_backend("numpy"):
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
                    [index % 2 for index in range(64)],
                    dtype=ts.uint8,
                    shape=(8, 8),
                )
                selected = ts.Variable(value_data)
                chosen = ts.where(condition, selected, right)
                ts.grad(chosen, selected, grad_outputs=ts.full((8, 8), 1.0))

                clipped = ts.Variable(value_data)
                clipped_output = ts.clip(clipped, 1.2, 1.6)
                ts.grad(
                    clipped_output,
                    clipped,
                    grad_outputs=ts.full((8, 8), 1.0),
                )

                for operation in (ts.minimum, ts.maximum):
                    selected = ts.Variable(value_data)
                    output = operation(selected, right)
                    ts.grad(
                        output,
                        selected,
                        grad_outputs=ts.full((8, 8), 1.0),
                    )

        self.assertEqual(
            [call.args[0] for call in reduction.call_args_list],
            ["std", "prod", "min", "max"],
        )
        self.assertEqual(
            [call.args[0] for call in reduction_gradient.call_args_list],
            ["std", "prod", "min", "max"],
        )
        self.assertEqual(
            [call.args[0] for call in arg_extremum.call_args_list],
            ["argmin", "argmax"],
        )
        self.assertEqual(comparison.call_count, 6)
        where.assert_called_once()
        where_gradient.assert_called_once()
        clip.assert_called_once()
        clip_gradient.assert_called_once()
        self.assertEqual(extremum.call_count, 2)
        self.assertEqual(extremum_gradient.call_count, 2)
    def test_remaining_reductions_and_selection_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                data = ts.Tensor(
                    [1.0 + (index % 8) / 10.0 for index in range(64)],
                    shape=(8, 8),
                )
                results = []
                for operation in (ts.std, ts.prod, ts.min, ts.max):
                    value = ts.Variable(data)
                    output = operation(value, axis=1)
                    gradient = ts.grad(
                        output,
                        value,
                        grad_outputs=ts.full((8,), 1.0),
                    )
                    results.extend((output.data, gradient))
                results.extend((
                    ts.argmin(data, axis=1),
                    ts.argmax(data, axis=1),
                    ts.greater_equal(data, 1.4),
                    ts.where(ts.greater(data, 1.4), data, 1.4),
                    ts.clip(data, 1.2, 1.6),
                    ts.minimum(data, 1.4),
                    ts.maximum(data, 1.4),
                ))
                return results

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)
    def test_reductions_dispatch_to_numpy(self):
        value = ts.Tensor([float(index + 1) for index in range(512)])
        with patch.object(
            numpy_backend,
            "reduction",
            wraps=numpy_backend.reduction,
        ) as reduction:
            with ts.use_backend("numpy"):
                ts.sum(value)
                ts.mean(value)
                ts.variance(value)
                ts.norm(value)

        self.assertEqual(
            [call.args[0] for call in reduction.call_args_list],
            ["sum", "mean", "variance", "norm"],
        )
    def test_axis_reductions_match_python_backend(self):
        value = ts.Tensor(
            [float(index + 1) for index in range(24)],
            shape=(2, 3, 4),
        )

        for operation in (ts.sum, ts.mean, ts.variance, ts.norm):
            with self.subTest(operation=operation.__name__):
                self.assertOperationParity(
                    lambda operation=operation: operation(
                        value,
                        axis=(0, 2),
                        keepdims=True,
                    )
                )
    def test_stable_reductions_fall_back_without_changing_results(self):
        value = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])
        smallest = ts.Tensor([5e-324, 5e-324])

        self.assertOperationParity(lambda: ts.sum(value))
        self.assertOperationParity(lambda: ts.mean(smallest))
        self.assertOperationParity(lambda: ts.variance(value))
        self.assertOperationParity(lambda: ts.norm(value))
    def test_broadcast_gradient_reductions_dispatch_to_numpy(self):
        left = ts.Variable(ts.full((64, 1), 2.0))
        right = ts.Variable(ts.full((1, 64), 3.0))
        with (
            patch.object(
                numpy_backend,
                "sum_to_shape",
                wraps=numpy_backend.sum_to_shape,
            ) as sum_kernel,
            patch.object(
                numpy_backend,
                "sum_products_to_shape",
                wraps=numpy_backend.sum_products_to_shape,
            ) as product_kernel,
        ):
            with ts.use_backend("numpy"):
                ts.grad(ts.sum(left + right), (left, right))
                ts.grad(ts.sum(left * right), (left, right))

        self.assertGreaterEqual(sum_kernel.call_count, 2)
        self.assertGreaterEqual(product_kernel.call_count, 2)
    def test_broadcast_gradient_reductions_match_python_backend(self):
        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.full((64, 1), 2.0))
                right = ts.Variable(ts.full((1, 64), 3.0))
                added = ts.grad(ts.sum(left + right), (left, right))
                multiplied = ts.grad(ts.sum(left * right), (left, right))
                return (
                    tuple(item.tolist() for item in added),
                    tuple(item.tolist() for item in multiplied),
                )

        self.assertEqual(gradients("numpy"), gradients("python"))
    def test_product_reduction_preserves_exact_cancellation(self):
        def gradient(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.Tensor([[1.0], [1.0]]))
                right = ts.Variable(
                    ts.Tensor([[1.0e308, -1.0e308]])
                )
                return ts.grad(ts.sum(left * right), left)

        self.assertEqual(
            gradient("numpy").tolist(),
            gradient("python").tolist(),
        )


if __name__ == "__main__":
    unittest.main()
