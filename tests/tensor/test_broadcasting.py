import unittest

import tensors as ts


class TensorBroadcastingTests(unittest.TestCase):
    def test_matrix_broadcasts_row_vector_over_rows(self):
        matrix = ts.Tensor([[1, 2], [3, 4], [5, 6]])
        row = ts.Tensor([10, 20])

        result = matrix + row

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.tolist(), [11.0, 22.0, 13.0, 24.0, 15.0, 26.0])

    def test_3d_tensor_broadcasts_matching_trailing_shape(self):
        tensor = ts.Tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])
        trailing = ts.Tensor([[10, 20], [30, 40]])

        result = tensor + trailing

        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.tolist(), [11.0, 22.0, 33.0, 44.0, 15.0, 26.0, 37.0, 48.0])

    def test_equal_shapes_do_not_change_result_shape(self):
        left = ts.Tensor([[1, 2], [3, 4]])
        right = ts.Tensor([[10, 20], [30, 40]])

        result = left + right

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [11.0, 22.0, 33.0, 44.0])

    def test_broadcasting_preserves_promoted_dtype(self):
        matrix = ts.Tensor([[1, 2], [3, 4]], dtype=ts.float32)
        row = ts.Tensor([10, 20], dtype=ts.float64)

        self.assertIs((matrix + row).dtype, ts.float64)

    def test_broadcasting_rejects_non_trailing_match(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            _ = tensor + ts.Tensor([[10], [20]])


if __name__ == "__main__":
    unittest.main()
