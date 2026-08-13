import unittest

import tensors as ts


class StackTests(unittest.TestCase):
    def test_stack(self):
        left = ts.Tensor([1, 2, 3])
        right = ts.Tensor([4, 5, 6])

        stacked_rows = ts.stack([left, right])
        self.assertEqual(stacked_rows.shape, (2, 3))
        self.assertEqual(stacked_rows.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        stacked_columns = ts.stack([left, right], axis=1)
        self.assertEqual(stacked_columns.shape, (3, 2))
        self.assertEqual(stacked_columns.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_stack_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            ts.stack([ts.Tensor([1, 2]), ts.Tensor([1, 2, 3])])

    def test_stack_with_lists(self):
        stacked = ts.stack([[1, 2], [3, 4]])

        self.assertEqual(stacked.shape, (2, 2))
        self.assertEqual(stacked.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stack_negative_axis(self):
        tensor = ts.Tensor([1, 2])
        stacked = ts.stack([tensor, tensor], axis=-1)

        self.assertEqual(stacked.shape, (2, 2))
        self.assertEqual(stacked.tolist(), [1.0, 1.0, 2.0, 2.0])

    def test_stack_rejects_empty_sequence(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ts.stack([])

    def test_stack_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.stack([ts.Tensor([1, 2])], axis=3)

    def test_stack_preserves_first_tensor_dtype(self):
        left = ts.Tensor([1, 2], dtype=ts.float32)
        right = ts.Tensor([3, 4], dtype=ts.float32)

        result = ts.stack([left, right])

        self.assertIs(result.dtype, ts.float32)

    def test_math_namespace_exposes_stack_operation_class(self):
        result = ts.math.Stack.forward([ts.Tensor([1, 2]), ts.Tensor([3, 4])])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stack_variables_is_differentiable(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])

        loss = ts.sum(ts.stack([left, right], axis=1) ** 2.0)
        ts.backward(loss)

        self.assertEqual(left.grad.tolist(), [2.0, 4.0])
        self.assertEqual(right.grad.tolist(), [6.0, 8.0])


if __name__ == "__main__":
    unittest.main()
