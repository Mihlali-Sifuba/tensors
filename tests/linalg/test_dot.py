import unittest

import tensors as ts


class LinalgTests(unittest.TestCase):
    def test_linalg_namespace_exposes_dot(self):
        left = ts.Tensor([[1.0, 2.0]])
        right = ts.Tensor([[3.0], [4.0]])

        self.assertEqual(ts.linalg.dot(left, right).tolist(), [11.0])

    def test_dot_preserves_promoted_dtype(self):
        left = ts.Tensor([[1.0, 2.0]], dtype=ts.float32)
        right = ts.Tensor([[3.0], [4.0]], dtype=ts.float64)

        result = ts.linalg.dot(left, right)

        self.assertIs(result.dtype, ts.float64)

    def test_dot_returns_scalar_for_two_vectors(self):
        result = ts.linalg.dot(ts.Tensor([1.0, 2.0]), ts.Tensor([3.0, 4.0]))

        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 11.0)

    def test_dot_recovers_from_temporary_overflow(self):
        left = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])
        right = ts.Tensor([1.0, 1.0, 1.0, 1.0])

        result = ts.dot(left, right)

        self.assertEqual(result.item(), 0.0)

    def test_dot_recovers_when_individual_products_overflow(self):
        left = ts.Tensor([1.0e308, 1.0e308, 1.0e-300])
        right = ts.Tensor([2.0, -2.0, 1.0])

        result = ts.dot(left, right)

        self.assertEqual(result.item(), 1.0e-300)

    def test_dot_supports_matrix_vector_and_vector_matrix_products(self):
        matrix = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        vector = ts.Tensor([5.0, 6.0])

        self.assertEqual(ts.linalg.dot(matrix, vector).tolist(), [17.0, 39.0])
        self.assertEqual(ts.linalg.dot(vector, matrix).tolist(), [23.0, 34.0])

    def test_matmul_supports_batched_matrix_products(self):
        matrices = ts.Tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        identity = ts.Tensor([[1.0, 0.0], [0.0, 1.0]])

        result = ts.matmul(matrices, identity)

        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(result.tolist(), matrices.tolist())

    def test_matmul_backpropagates_through_broadcast_batch_dimensions(self):
        left = ts.Variable(ts.Tensor([1.0, 2.0], shape=(1, 1, 2)))
        right = ts.Variable(ts.Tensor([3.0, 4.0, 5.0, 6.0], shape=(2, 2, 1)))
        loss = ts.sum(left @ right)

        ts.backward(loss)

        self.assertEqual(left.grad.shape, (1, 1, 2))
        self.assertEqual(left.grad.tolist(), [8.0, 10.0])
        self.assertEqual(right.grad.shape, (2, 2, 1))
        self.assertEqual(right.grad.tolist(), [1.0, 2.0, 1.0, 2.0])

    def test_matmul_gradient_recovers_from_temporary_overflow(self):
        left = ts.Tensor(
            [1.0e308, 1.0e308, -1.0e308, -1.0e308],
            shape=(4, 1),
        )
        right = ts.Variable([1.0])

        ts.backward(ts.sum(left @ right))

        self.assertEqual(right.grad.tolist(), [0.0])

    def test_dot_rejects_scalar_inputs(self):
        scalar = ts.Tensor([1.0], shape=())

        with self.assertRaisesRegex(ValueError, "at least one dimension"):
            ts.linalg.dot(scalar, ts.Tensor([1.0]))

    def test_dot_rejects_inner_dimension_mismatch(self):
        left = ts.Tensor([[1.0, 2.0, 3.0]])
        right = ts.Tensor([[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            ts.linalg.dot(left, right)


if __name__ == "__main__":
    unittest.main()
