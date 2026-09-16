"""ReLU, sigmoid, and softplus."""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class ActivationTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_relu_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([-2.0, 0.0, 3.0])

        result = ts.relu(tensor)

        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.tolist(), [0.0, 0.0, 3.0])

    def test_relu_propagates_nan_in_values_and_gradients(self):
        value = ts.Variable([math.nan])

        result = ts.relu(value)
        gradient = ts.grad(result, value)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(gradient.item()))

    def test_relu_preserves_input_dtype(self):
        self.assertIs(ts.relu(ts.Tensor([-1, 2], dtype=ts.int32)).dtype, ts.int32)
        self.assertIs(ts.relu(ts.Tensor([-1, 2], dtype=ts.float32)).dtype, ts.float32)

    def test_sigmoid_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 2.0])

        result = ts.sigmoid(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.5)
        self.assertAlmostEqual(result.tolist()[1], 1.0 / (1.0 + math.exp(-2.0)))

    def test_softplus_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 2.0])

        result = ts.softplus(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], math.log(2.0))
        self.assertAlmostEqual(result.tolist()[1], math.log1p(math.exp(2.0)))

    def test_relu_backward(self):
        x = ts.Variable([-1.0, 0.0, 2.0])
        loss = ts.sum(ts.relu(x))

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 1.0])

    def test_sigmoid_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.sigmoid(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.25)

    def test_softplus_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.softplus(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
