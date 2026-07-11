import unittest

import tensors as ts


class TensorOpsTests(unittest.TestCase):
    def test_operations_preserve_float32(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        self.assertIs((tensor + 1).dtype, ts.float32)
        self.assertIs(tensor[1:3].dtype, ts.float32)
        self.assertIs(ts.reshape(tensor, (2, 2)).dtype, ts.float32)
        self.assertIs(ts.transpose(ts.reshape(tensor, (2, 2))).dtype, ts.float32)

    def test_integer_division_promotes_to_float(self):
        result = ts.Tensor([2, 4], dtype=ts.int32) / 2

        self.assertIs(result.dtype, ts.float64)
        self.assertEqual(result.tolist(), [1.0, 2.0])

    def test_reverse_scalar_operators(self):
        tensor = ts.Tensor([1, 2])

        self.assertEqual((2 + tensor).tolist(), [3.0, 4.0])
        self.assertEqual((2 - tensor).tolist(), [1.0, 0.0])
        self.assertEqual((2 * tensor).tolist(), [2.0, 4.0])
        self.assertEqual((2 / tensor).tolist(), [2.0, 1.0])

    def test_package_level_operation_aliases(self):
        left = ts.Tensor([1, 2])
        right = ts.Tensor([3, 4])

        self.assertEqual(ts.add(left, right).tolist(), [4.0, 6.0])
        self.assertEqual(ts.subtract(right, left).tolist(), [2.0, 2.0])
        self.assertEqual(ts.multiply(left, right).tolist(), [3.0, 8.0])
        self.assertEqual(ts.divide(right, left).tolist(), [3.0, 2.0])

    def test_tensor_dtype_promotion_prefers_wider_float(self):
        left = ts.Tensor([1, 2], dtype=ts.float32)
        right = ts.Tensor([3, 4], dtype=ts.float64)

        self.assertIs((left + right).dtype, ts.float64)

    def test_tensor_dtype_promotion_prefers_wider_integer(self):
        left = ts.Tensor([1, 2], dtype=ts.int16)
        right = ts.Tensor([3, 4], dtype=ts.int64)

        self.assertIs((left + right).dtype, ts.int64)

    def test_negating_uint8_promotes_to_signed_dtype(self):
        result = -ts.Tensor([1, 2], dtype=ts.uint8)

        self.assertIs(result.dtype, ts.int16)
        self.assertEqual(result.tolist(), [-1, -2])

    def test_division_by_zero_is_rejected(self):
        with self.assertRaises(ZeroDivisionError):
            _ = ts.Tensor([1, 2]) / 0

        with self.assertRaises(ZeroDivisionError):
            _ = ts.Tensor([1, 2]) / ts.Tensor([1, 0])

    def test_broadcast_add(self):
        left = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        right = ts.Tensor([10, 20, 30])

        result = left + right

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.tolist(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])

    def test_broadcast_rejects_incompatible_shapes(self):
        left = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            _ = left + ts.Tensor([1, 2])

    def test_broadcast_sub_mul_div(self):
        left = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        right = ts.Tensor([10, 20, 30])

        self.assertEqual((left - right).tolist(), [-9.0, -18.0, -27.0, -6.0, -15.0, -24.0])
        self.assertEqual((left * right).tolist(), [10.0, 40.0, 90.0, 40.0, 100.0, 180.0])
        self.assertEqual((left / right).tolist(), [0.1, 0.1, 0.1, 0.4, 0.25, 0.2])


if __name__ == "__main__":
    unittest.main()
