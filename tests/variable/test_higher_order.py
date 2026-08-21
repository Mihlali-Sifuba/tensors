import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class HigherOrderDerivativeTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_grad_builds_a_graph_for_second_and_third_derivatives(self):
        value = ts.Variable([3.0])
        loss = ts.sum(value ** 3.0)

        first = ts.grad(loss, value, create_graph=True)
        second = ts.grad(first, value, create_graph=True)
        third = ts.grad(second, value)

        self.assertIsInstance(first, ts.Variable)
        self.assertIsInstance(second, ts.Variable)
        self.assertEqual(first.data.tolist(), [27.0])
        self.assertEqual(second.data.tolist(), [18.0])
        self.assertEqual(third.tolist(), [6.0])

    def test_exp_second_derivative_is_exp(self):
        value = ts.Variable([1.0])
        output = ts.exp(value)

        first = ts.grad(output, value, create_graph=True)
        second = ts.grad(first, value)

        self.assertAlmostEqual(first.data.item(), math.e)
        self.assertAlmostEqual(second.item(), math.e)

    def test_backward_create_graph_retains_a_differentiable_gradient(self):
        value = ts.Variable([2.0])
        output = value ** 3.0

        ts.backward(output, create_graph=True)

        self.assertIsInstance(value.grad, ts.Variable)
        self.assertEqual(value.grad.data.tolist(), [12.0])
        self.assertEqual(ts.grad(value.grad, value).tolist(), [12.0])

    def test_grad_returns_gradients_for_multiple_inputs(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = left * right

        left_gradient, right_gradient = ts.grad(output, (left, right))

        self.assertEqual(left_gradient.tolist(), [3.0])
        self.assertEqual(right_gradient.tolist(), [2.0])

    def test_grad_outputs_weights_each_output_element(self):
        value = ts.Variable([2.0, 3.0])
        output = value ** 2.0

        gradient = ts.grad(output, value, grad_outputs=ts.Tensor([4.0, 5.0]))

        self.assertEqual(gradient.tolist(), [16.0, 30.0])

    def test_grad_outputs_must_match_output_shape(self):
        value = ts.Variable([2.0, 3.0])
        output = value ** 2.0

        with self.assertRaisesRegex(ValueError, "Gradient shape"):
            ts.grad(output, value, grad_outputs=ts.Tensor([1.0]))

    def test_broadcast_gradient_can_be_differentiated(self):
        value = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
        bias = ts.Variable([0.5, -0.5])
        loss = ts.sum((value + bias) ** 2.0)

        first = ts.grad(loss, bias, create_graph=True)
        second = ts.grad(first, bias, grad_outputs=ts.Tensor([1.0, 1.0]))

        self.assertEqual(first.data.tolist(), [10.0, 10.0])
        self.assertEqual(second.tolist(), [4.0, 4.0])

    def test_batched_matmul_gradient_can_be_differentiated(self):
        left = ts.Variable([[[1.0, 2.0]], [[3.0, 4.0]]])
        right = ts.Variable([[[1.0], [2.0]]])
        loss = ts.sum((left @ right) ** 2.0)

        first = ts.grad(loss, left, create_graph=True)
        second = ts.grad(
            first,
            left,
            grad_outputs=ts.Tensor([[[1.0, 1.0]], [[1.0, 1.0]]]),
        )

        self.assertEqual(second.shape, left.shape)
        self.assertEqual(second.tolist(), [6.0, 12.0, 6.0, 12.0])

    def test_vector_dot_gradient_can_be_differentiated(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])

        left_gradient = ts.grad(
            ts.dot(left, right),
            left,
            create_graph=True,
        )
        mixed_gradient = ts.grad(ts.sum(left_gradient), right)

        self.assertEqual(left_gradient.data.tolist(), [3.0, 4.0])
        self.assertEqual(mixed_gradient.tolist(), [1.0, 1.0])

    def test_matrix_vector_gradients_can_be_differentiated(self):
        matrix = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
        vector = ts.Variable([5.0, 6.0])

        matrix_gradient = ts.grad(
            ts.sum(matrix @ vector),
            matrix,
            create_graph=True,
        )
        mixed_gradient = ts.grad(ts.sum(matrix_gradient), vector)

        self.assertEqual(matrix_gradient.data.tolist(), [5.0, 6.0, 5.0, 6.0])
        self.assertEqual(mixed_gradient.tolist(), [2.0, 2.0])

    def test_vector_matrix_gradients_can_be_differentiated(self):
        vector = ts.Variable([1.0, 2.0])
        matrix = ts.Variable([[3.0, 4.0], [5.0, 6.0]])

        vector_gradient = ts.grad(
            ts.sum(vector @ matrix),
            vector,
            create_graph=True,
        )
        mixed_gradient = ts.grad(ts.sum(vector_gradient), matrix)

        self.assertEqual(vector_gradient.data.tolist(), [7.0, 11.0])
        self.assertEqual(mixed_gradient.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_singleton_standard_deviation_has_zero_higher_derivatives(self):
        value = ts.Variable([4.0])

        first = ts.grad(ts.std(value), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [0.0])
        self.assertEqual(second.tolist(), [0.0])

    def test_softmax_gradient_can_be_differentiated(self):
        value = ts.Variable([[0.2, -0.4, 0.7]])
        loss = ts.sum(ts.softmax(value, axis=1) ** 2.0)

        first = ts.grad(loss, value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([[1.0, 1.0, 1.0]]),
        )

        for item in second.tolist():
            self.assertAlmostEqual(item, 0.0, places=12)

    def test_axis_std_gradient_can_be_differentiated(self):
        value = ts.Variable([[1.0, 2.0, 4.0], [2.0, 5.0, 9.0]])
        loss = ts.sum(ts.std(value, axis=1))

        first = ts.grad(loss, value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        )

        for item in second.tolist():
            self.assertAlmostEqual(item, 0.0, places=12)

    def test_concat_and_stack_gradients_can_be_differentiated(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])
        concatenated = ts.concat([left, right])
        stacked = ts.stack([left, right])
        loss = ts.sum(concatenated ** 2.0) + ts.sum(stacked ** 2.0)

        first = ts.grad(loss, left, create_graph=True)
        second = ts.grad(
            first,
            left,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [4.0, 8.0])
        self.assertEqual(second.tolist(), [4.0, 4.0])

    def test_shared_gradient_terms_are_stable_and_differentiable(self):
        value = ts.Variable([1.0])
        seed = ts.Variable([1.0e308, 1.0e308, -1.0e308, -1.0e308])
        output = ts.concat([value, value, value, value])

        gradient = ts.grad(
            output,
            value,
            grad_outputs=seed,
            create_graph=True,
        )
        seed_gradient = ts.grad(gradient, seed)

        self.assertEqual(gradient.data.tolist(), [0.0])
        self.assertEqual(seed_gradient.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_slice_gradient_can_be_differentiated(self):
        value = ts.Variable([1.0, 2.0, 3.0])
        loss = ts.sum(value[1:] ** 3.0)

        first = ts.grad(loss, value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([1.0, 1.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [0.0, 12.0, 27.0])
        self.assertEqual(second.tolist(), [0.0, 12.0, 18.0])


if __name__ == "__main__":
    unittest.main()
