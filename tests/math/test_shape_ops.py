import unittest

import tensors as ts


class ShapeOpsTests(unittest.TestCase):
    def test_reshape_to_scalar_shape(self):
        result = ts.reshape(ts.Tensor([7.0]), ())

        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 7.0)

    def test_reshape_to_zero_sized_shape(self):
        result = ts.reshape(ts.Tensor([]), (0, 3))

        self.assertEqual(result.shape, (0, 3))
        self.assertEqual(result.tolist(), [])

    def test_reshape_accepts_list_shape(self):
        result = ts.reshape(ts.Tensor([1, 2, 3, 4]), [2, 2])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_reshape_returns_independent_tensor(self):
        original = ts.Tensor([1, 2, 3, 4])

        reshaped = ts.reshape(original, (2, 2))
        reshaped[0, 0] = 99

        self.assertEqual(original.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_transpose_square_matrix(self):
        result = ts.transpose(ts.Tensor([[1, 2], [3, 4]]))

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 3.0, 2.0, 4.0])

    def test_transpose_rejects_variable_until_differentiable_rule_exists(self):
        with self.assertRaises(NotImplementedError):
            ts.transpose(ts.Variable([[1.0, 2.0]]))


if __name__ == "__main__":
    unittest.main()
