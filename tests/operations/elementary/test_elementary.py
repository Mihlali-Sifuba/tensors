"""Exponential, logarithm, and square root."""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class ElementaryFunctionTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_exp_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 1.0])

        result = ts.exp(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 1.0)
        self.assertAlmostEqual(result.tolist()[1], math.e)

    def test_exp_returns_infinity_when_the_result_overflows(self):
        value = ts.Variable([1000.0])

        result = ts.exp(value)
        gradient = ts.grad(result, value)

        self.assertEqual(result.data.tolist(), [math.inf])
        self.assertEqual(gradient.tolist(), [math.inf])

    def test_log_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([1.0, math.e])

        result = ts.log(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.0)
        self.assertAlmostEqual(result.tolist()[1], 1.0)

    def test_sqrt_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([4.0, 9.0])

        result = ts.sqrt(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [2.0, 3.0])

    def test_log_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ts.log(ts.Tensor([0.0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            ts.log(ts.Tensor([-1.0]))

    def test_sqrt_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ts.sqrt(ts.Tensor([-1.0]))

    def test_sqrt_gradient_rejects_zero_where_derivative_is_undefined(self):
        value = ts.Variable([0.0])

        with self.assertRaisesRegex(ValueError, "undefined at zero"):
            ts.backward(ts.sqrt(value))

    def test_exp_backward(self):
        x = ts.Variable([0.0, 1.0])
        loss = ts.sum(ts.exp(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.tolist()[0], 1.0)
        self.assertAlmostEqual(x.grad.tolist()[1], math.e)

    def test_log_backward(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.sum(ts.log(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.tolist()[0], 1.0)
        self.assertAlmostEqual(x.grad.tolist()[1], 0.5)

    def test_sqrt_backward(self):
        x = ts.Variable([4.0])
        loss = ts.sum(ts.sqrt(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.25)


if __name__ == "__main__":
    unittest.main()
