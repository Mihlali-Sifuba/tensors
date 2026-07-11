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

    def test_clone_creates_independent_copy(self):
        t = ts.Tensor([1, 2, 3])
        c = t.clone()
        self.assertEqual(c.tolist(), [1.0, 2.0, 3.0])
        c._data[0] = 99
        self.assertEqual(t.tolist(), [1.0, 2.0, 3.0])

    def test_astype_converts_dtype(self):
        t = ts.Tensor([1, 2, 3])
        self.assertIs(t.astype(ts.float32).dtype, ts.float32)
        self.assertIs(t.astype(ts.int32).dtype, ts.int32)
        self.assertEqual(t.astype(ts.int32).tolist(), [1, 2, 3])

    def test_len_returns_first_dimension(self):
        self.assertEqual(len(ts.Tensor([1, 2, 3])), 3)
        self.assertEqual(len(ts.Tensor([[1, 2], [3, 4], [5, 6]])), 3)

    def test_broadcast_add(self):
        a = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        b = ts.Tensor([10, 20, 30])
        result = a + b
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.tolist(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])

    def test_broadcast_rejects_incompatible_shapes(self):
        a = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            _ = a + ts.Tensor([1, 2])

    def test_sum_axis(self):
        m = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(ts.sum(m, axis=0).tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(ts.sum(m, axis=1).tolist(), [6.0, 15.0])

    def test_mean_axis(self):
        m = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(ts.mean(m, axis=0).tolist(), [2.5, 3.5, 4.5])
        self.assertEqual(ts.mean(m, axis=1).tolist(), [2.0, 5.0])

    def test_sum_keepdims(self):
        m = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(ts.sum(m, axis=1, keepdims=True).shape, (2, 1))

    def test_stack(self):
        a = ts.Tensor([1, 2, 3])
        b = ts.Tensor([4, 5, 6])
        s0 = ts.stack([a, b])
        self.assertEqual(s0.shape, (2, 3))
        self.assertEqual(s0.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        s1 = ts.stack([a, b], axis=1)
        self.assertEqual(s1.shape, (3, 2))
        self.assertEqual(s1.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_stack_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            ts.stack([ts.Tensor([1, 2]), ts.Tensor([1, 2, 3])])

    def test_stack_with_lists(self):
        s = ts.stack([[1, 2], [3, 4]])
        self.assertEqual(s.shape, (2, 2))
        self.assertEqual(s.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stack_negative_axis(self):
        a = ts.Tensor([1, 2])
        s = ts.stack([a, a], axis=-1)
        self.assertEqual(s.shape, (2, 2))
        self.assertEqual(s.tolist(), [1.0, 1.0, 2.0, 2.0])

    def test_sum_negative_axis(self):
        m = ts.Tensor([[1, 2], [3, 4]])
        self.assertEqual(ts.sum(m, axis=-1).tolist(), [3.0, 7.0])

    def test_mean_keepdims(self):
        m = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(ts.mean(m, axis=0, keepdims=True).shape, (1, 3))

    def test_broadcast_sub_mul_div(self):
        a = ts.Tensor([[1, 2, 3], [4, 5, 6]])
        b = ts.Tensor([10, 20, 30])
        self.assertEqual((a - b).tolist(), [-9.0, -18.0, -27.0, -6.0, -15.0, -24.0])
        self.assertEqual((a * b).tolist(), [10.0, 40.0, 90.0, 40.0, 100.0, 180.0])
        self.assertEqual((a / b).tolist(), [0.1, 0.1, 0.1, 0.4, 0.25, 0.2])

    def test_clone_preserves_dtype(self):
        t = ts.Tensor([1, 2], dtype=ts.float32)
        self.assertIs(t.clone().dtype, ts.float32)

    def test_astype_with_string_typecode(self):
        t = ts.Tensor([1, 2, 3])
        self.assertIs(t.astype('f').dtype, ts.float32)

    def test_reshape_error_on_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Cannot reshape"):
            ts.reshape(ts.Tensor([1, 2, 3]), (2,))


if __name__ == "__main__":
    unittest.main()
