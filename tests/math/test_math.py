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

    def test_sum_axis(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.sum(matrix, axis=0).tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(ts.sum(matrix, axis=1).tolist(), [6.0, 15.0])

    def test_mean_axis(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.mean(matrix, axis=0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(ts.mean(matrix, axis=1).tolist(), [2.0, 5.0])

    def test_sum_keepdims(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.sum(matrix, axis=1, keepdims=True).shape, (2, 1))

    def test_sum_negative_axis(self):
        matrix = ts.Tensor([[1, 2], [3, 4]])

        self.assertEqual(ts.sum(matrix, axis=-1).tolist(), [3.0, 7.0])

    def test_sum_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.sum(ts.Tensor([[1, 2]]), axis=2)

    def test_mean_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.mean(ts.Tensor([[1, 2]]), axis=2)

    def test_mean_keepdims(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(ts.mean(matrix, axis=0, keepdims=True).shape, (1, 3))

    def test_empty_mean_returns_zero_scalar(self):
        result = ts.mean(ts.Tensor([]))

        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.tolist(), [0.0])

    def test_std_accepts_plain_lists(self):
        result = ts.std([1.0, 2.0, 3.0])

        self.assertAlmostEqual(result.item(), 0.816496580927726)

    def test_reshape_error_on_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Cannot reshape"):
            ts.reshape(ts.Tensor([1, 2, 3]), (2,))

    def test_reshape_preserves_dtype(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        result = ts.reshape(tensor, (2, 2))

        self.assertEqual(result.shape, (2, 2))
        self.assertIs(result.dtype, ts.float32)


if __name__ == "__main__":
    unittest.main()
