import unittest

import tensors as ts


class SelectionTests(unittest.TestCase):
    def test_where_broadcasts_condition_and_values(self):
        condition = ts.Tensor([[1], [0]], dtype=ts.uint8)
        left = ts.Tensor([1.0, 2.0])
        right = ts.Tensor([[3.0], [4.0]])

        result = ts.where(condition, left, right)

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 4.0, 4.0])

    def test_where_promotes_integer_tensor_and_float_scalar(self):
        result = ts.where([1, 0], ts.Tensor([1, 2], dtype=ts.int16), 2.5)

        self.assertIs(result.dtype, ts.float64)
        self.assertEqual(result.tolist(), [1.0, 2.5])

    def test_where_routes_gradients_and_reduces_broadcast_axes(self):
        left = ts.Variable([[1.0], [2.0]])
        right = ts.Variable([3.0, 4.0])
        condition = ts.Tensor([[1, 0], [0, 1]], dtype=ts.uint8)

        ts.backward(ts.sum(ts.where(condition, left, right)))

        self.assertEqual(left.grad.tolist(), [1.0, 1.0])
        self.assertEqual(right.grad.tolist(), [1.0, 1.0])

    def test_where_gradient_has_zero_higher_derivative(self):
        value = ts.Variable([1.0, 2.0, 3.0])
        output = ts.where([1, 0, 1], value, -value)

        first = ts.grad(ts.sum(output), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [1.0, -1.0, 1.0])
        self.assertEqual(second.tolist(), [0.0, 0.0, 0.0])

    def test_where_rejects_a_trainable_condition(self):
        condition = ts.Variable([1.0, 0.0])

        with self.assertRaisesRegex(TypeError, "condition cannot require"):
            ts.where(condition, [1.0, 2.0], [3.0, 4.0])

    def test_where_tracks_frozen_condition_mutation_and_recomputation(self):
        condition = ts.Variable(
            ts.Tensor([1, 0], dtype=ts.uint8),
            requires_grad=False,
        )
        value = ts.Variable([1.0, 2.0])
        output = ts.where(condition, value, -value)
        computation = ts.graph.Computation(output)

        condition.data[0] = 0
        with self.assertRaisesRegex(RuntimeError, "modified after"):
            ts.backward(ts.sum(output))

        recomputed = computation.forward()
        self.assertEqual(recomputed.tolist(), [-1.0, -2.0])

    def test_elementwise_extrema_broadcast_and_promote_dtype(self):
        left = ts.Tensor([[1.0], [4.0]], dtype=ts.float32)
        right = ts.Tensor([2, 3], dtype=ts.int16)

        minimum = ts.minimum(left, right)
        maximum = ts.maximum(left, right)

        self.assertIs(minimum.dtype, ts.float32)
        self.assertIs(maximum.dtype, ts.float32)
        self.assertEqual(minimum.tolist(), [1.0, 1.0, 2.0, 3.0])
        self.assertEqual(maximum.tolist(), [2.0, 3.0, 4.0, 4.0])

    def test_elementwise_extrema_promote_integer_tensor_and_float_scalar(self):
        result = ts.maximum(ts.Tensor([1, 3], dtype=ts.int16), 2.5)

        self.assertIs(result.dtype, ts.float64)
        self.assertEqual(result.tolist(), [2.5, 3.0])

    def test_elementwise_extrema_split_first_gradient_at_ties(self):
        left = ts.Variable([1.0, 2.0, 4.0])
        right = ts.Variable([2.0, 2.0, 3.0])

        ts.backward(ts.sum(ts.maximum(left, right)))

        self.assertEqual(left.grad.tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(right.grad.tolist(), [1.0, 0.5, 0.0])

    def test_elementwise_extrema_have_zero_higher_derivatives_away_from_ties(self):
        left = ts.Variable([1.0, 4.0])
        right = ts.Variable([2.0, 3.0])

        first = ts.grad(ts.sum(ts.minimum(left, right)), left, create_graph=True)
        second = ts.grad(ts.sum(first), left)

        self.assertEqual(first.data.tolist(), [1.0, 0.0])
        self.assertEqual(second.tolist(), [0.0, 0.0])

    def test_elementwise_extrema_reject_higher_derivatives_at_ties(self):
        value = ts.Variable([1.0])

        with self.assertRaisesRegex(ValueError, "undefined at ties"):
            ts.grad(ts.maximum(value, [1.0]), value, create_graph=True)


if __name__ == "__main__":
    unittest.main()
