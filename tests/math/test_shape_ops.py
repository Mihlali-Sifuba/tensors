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

    def test_reshape_keeps_variable_history_differentiable(self):
        variable = ts.Variable([1.0, 2.0, 3.0, 4.0])

        reshaped = ts.reshape(variable, (2, 2))
        ts.backward(ts.sum(reshaped))

        self.assertEqual(reshaped.shape, (2, 2))
        self.assertEqual(variable.grad.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_transpose_square_matrix(self):
        result = ts.transpose(ts.Tensor([[1, 2], [3, 4]]))

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 3.0, 2.0, 4.0])

    def test_transpose_keeps_variable_history_differentiable(self):
        variable = ts.Variable([[1.0, 2.0], [3.0, 4.0]])

        transposed = ts.transpose(variable)
        ts.backward(ts.sum(transposed))

        self.assertEqual(transposed.data.tolist(), [1.0, 3.0, 2.0, 4.0])
        self.assertEqual(transposed.shape, (2, 2))
        self.assertEqual(variable.grad.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_transpose_swaps_final_axes_for_batched_variable(self):
        variable = ts.Variable(ts.Tensor([1.0, 2.0, 3.0, 4.0], shape=(2, 1, 2)))

        transposed = ts.transpose(variable)

        self.assertEqual(transposed.shape, (2, 2, 1))
        self.assertEqual(transposed.data.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_transpose_accepts_an_axis_permutation(self):
        variable = ts.Variable(
            ts.Tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                shape=(1, 2, 3),
            )
        )

        transposed = ts.transpose(variable, axes=(2, 0, 1))
        ts.backward(ts.sum(transposed))

        self.assertEqual(transposed.shape, (3, 1, 2))
        self.assertEqual(transposed.data.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        self.assertEqual(variable.grad.tolist(), [1.0] * 6)


if __name__ == "__main__":
    unittest.main()
