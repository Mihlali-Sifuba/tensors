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


if __name__ == "__main__":
    unittest.main()
