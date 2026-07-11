import unittest

import tensors as ts


class TensorTests(unittest.TestCase):
    def test_operations_preserve_float32(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        self.assertIs((tensor + 1).dtype, ts.float32)
        self.assertIs(tensor[1:3].dtype, ts.float32)
        self.assertIs(ts.reshape(tensor, (2, 2)).dtype, ts.float32)
        self.assertIs(ts.transpose(ts.reshape(tensor, (2, 2))).dtype, ts.float32)

    def test_reverse_slice_uses_python_semantics(self):
        tensor = ts.Tensor([1, 2, 3, 4])

        self.assertEqual(tensor[::-1].tolist(), [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(tensor[3:0:-2].tolist(), [4.0, 2.0])

    def test_single_index_selects_first_dimension(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        self.assertEqual(tensor[0].shape, (2,))
        self.assertEqual(tensor[0].tolist(), [1.0, 2.0])

    def test_ragged_nested_lists_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ragged"):
            ts.Tensor([[1], [2, 3], []])

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid shape"):
            ts.Tensor([1, 2], shape=(-1, 2))

    def test_integer_division_promotes_to_float(self):
        result = ts.Tensor([2, 4], dtype=ts.int32) / 2

        self.assertIs(result.dtype, ts.float64)
        self.assertEqual(result.tolist(), [1.0, 2.0])

    def test_transpose_reports_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            ts.transpose(ts.Tensor([1, 2]))

    def test_reverse_scalar_operators(self):
        tensor = ts.Tensor([1, 2])

        self.assertEqual((2 + tensor).tolist(), [3.0, 4.0])
        self.assertEqual((2 - tensor).tolist(), [1.0, 0.0])
        self.assertEqual((2 / tensor).tolist(), [2.0, 1.0])

    def test_math_namespace_returns_tensor_scalars(self):
        tensor = ts.Tensor([1.0, 2.0, 3.0])

        self.assertEqual(ts.math.sum(tensor).tolist(), [6.0])
        self.assertEqual(ts.mean(tensor).tolist(), [2.0])

    def test_linalg_namespace_exposes_dot(self):
        left = ts.Tensor([[1.0, 2.0]])
        right = ts.Tensor([[3.0], [4.0]])

        self.assertEqual(ts.linalg.dot(left, right).tolist(), [11.0])

    def test_scalar_tensor_supports_item_and_numeric_formatting(self):
        value = ts.std(ts.Tensor([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(value.item(), 0.816496580927726)
        self.assertEqual(f"{value:.4f}", "0.8165")

    def test_item_rejects_multi_element_tensor(self):
        with self.assertRaisesRegex(ValueError, "one element"):
            ts.Tensor([1.0, 2.0]).item()


if __name__ == "__main__":
    unittest.main()
