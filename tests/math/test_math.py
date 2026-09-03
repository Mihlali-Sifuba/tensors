import math
import unittest

import tensors as ts


class MathTests(unittest.TestCase):
    def test_math_namespace_returns_tensor_scalars(self):
        tensor = ts.Tensor([1.0, 2.0, 3.0])

        self.assertEqual(ts.math.sum(tensor).tolist(), [6.0])
        self.assertEqual(ts.mean(tensor).tolist(), [2.0])

    def test_transpose_reports_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            ts.transpose(ts.Tensor([1, 2]))

    def test_transpose_swaps_2d_axes_and_preserves_dtype(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ts.float32)

        result = ts.transpose(matrix)

        self.assertEqual(result.shape, (3, 2))
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_transpose_rejects_boolean_axes(self):
        with self.assertRaisesRegex(TypeError, "only integers"):
            ts.transpose(
                ts.Tensor([[1.0, 2.0], [3.0, 4.0]]),
                axes=(True, False),
            )

    def test_sum_axis(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.sum(matrix, axis=0).tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(ts.sum(matrix, axis=1).tolist(), [6.0, 15.0])

    def test_sum_recovers_from_temporary_overflow(self):
        tensor = ts.Tensor(
            [
                1.0e308, 1.0e308, -1.0e308, -1.0e308, 1.0e-300,
                1.0e308, 1.0e308, 0.0, 0.0, 0.0,
            ],
            shape=(2, 5),
        )

        result = ts.sum(tensor, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result._data[0], 1.0e-300)
        self.assertEqual(result._data[1], math.inf)

    def test_mean_axis(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.mean(matrix, axis=0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(ts.mean(matrix, axis=1).tolist(), [2.0, 5.0])

    def test_mean_avoids_intermediate_overflow(self):
        matrix = ts.Tensor(
            [1.0e308, 1.0e308, 1.0e308, -1.0e308],
            shape=(2, 2),
        )

        result = ts.mean(matrix, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [1.0e308, 0.0])

    def test_mean_does_not_underflow_individual_terms(self):
        smallest = math.ulp(0.0)

        result = ts.mean([smallest, smallest])

        self.assertEqual(result.item(), smallest)

    def test_mean_of_opposite_infinities_is_nan(self):
        result = ts.mean([math.inf, -math.inf])

        self.assertTrue(math.isnan(result.item()))

    def test_sum_keepdims(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.sum(matrix, axis=1, keepdims=True).shape, (2, 1))

    def test_sum_negative_axis(self):
        matrix = ts.Tensor([[1, 2], [3, 4]])

        self.assertEqual(ts.sum(matrix, axis=-1).tolist(), [3.0, 7.0])

    def test_sum_accepts_multiple_axes(self):
        tensor = ts.Tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape=(2, 2, 2),
        )

        result = ts.sum(tensor, axis=(0, 2))

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [14.0, 22.0])

    def test_mean_accepts_multiple_negative_axes_and_keepdims(self):
        tensor = ts.Tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            shape=(2, 2, 2),
        )

        result = ts.mean(tensor, axis=(0, -1), keepdims=True)

        self.assertEqual(result.shape, (1, 2, 1))
        self.assertEqual(result.tolist(), [3.5, 5.5])

    def test_reductions_reject_duplicate_axes(self):
        with self.assertRaisesRegex(ValueError, "Duplicate axis"):
            ts.sum(ts.Tensor([[1.0]]), axis=(0, -2))

    def test_sum_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.sum(ts.Tensor([[1, 2]]), axis=2)

    def test_mean_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.mean(ts.Tensor([[1, 2]]), axis=2)

    def test_mean_keepdims(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.mean(matrix, axis=0, keepdims=True).shape, (1, 3))

    def test_empty_mean_returns_nan_scalar(self):
        result = ts.mean(ts.Tensor([]))

        self.assertEqual(result.shape, (1,))
        self.assertTrue(math.isnan(result.item()))

    def test_std_accepts_plain_lists(self):
        result = ts.std([1.0, 2.0, 3.0])

        self.assertAlmostEqual(result.item(), 0.816496580927726)

    def test_std_is_axis_aware_and_supports_keepdims(self):
        matrix = ts.Tensor([[1.0, 3.0], [2.0, 6.0]])

        result = ts.std(matrix, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [1.0, 2.0])

    def test_std_avoids_intermediate_overflow(self):
        matrix = ts.Tensor(
            [1.0e308, 1.0e308, 1.0e308, -1.0e308],
            shape=(2, 2),
        )

        result = ts.std(matrix, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [0.0, 1.0e308])

    def test_std_of_identical_subnormal_values_is_zero(self):
        smallest = math.ulp(0.0)
        value = ts.Variable([smallest, smallest])

        result = ts.std(value)
        ts.backward(result)

        self.assertEqual(result.data.item(), 0.0)
        self.assertEqual(value.grad.tolist(), [0.0, 0.0])

    def test_large_std_has_finite_gradient(self):
        value = ts.Variable([1.0e308, -1.0e308])

        ts.backward(ts.std(value))

        self.assertEqual(value.grad.tolist(), [0.5, -0.5])

    def test_std_handles_an_overflowing_centering_difference(self):
        count = 1001
        value = ts.Tensor([-1.0e308] * (count - 1) + [1.0e308])

        result = ts.std(value).item()
        expected = 1.0e308 * (2.0 * math.sqrt(count - 1) / count)

        self.assertAlmostEqual(result / expected, 1.0, places=14)

    def test_axis_std_is_differentiable(self):
        matrix = ts.Variable([[1.0, 3.0], [2.0, 6.0]])

        ts.backward(ts.sum(ts.std(matrix, axis=1)))

        self.assertEqual(matrix.grad.tolist(), [-0.5, 0.5, -0.5, 0.5])

    def test_math_namespace_exposes_std_operation_class(self):
        result = ts.math.Std().forward(ts.Tensor([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(result.item(), 0.816496580927726)

    def test_reshape_error_on_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Cannot reshape"):
            ts.reshape(ts.Tensor([1, 2, 3]), (2,))

    def test_reshape_preserves_dtype(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        result = ts.reshape(tensor, (2, 2))

        self.assertEqual(result.shape, (2, 2))
        self.assertIs(result.dtype, ts.float32)

    def test_math_namespace_exposes_reshape_operation_class(self):
        result = ts.math.Reshape(shape=(2, 2)).forward(ts.Tensor([1, 2, 3, 4]))

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
