"""The hyperbolic tangent, reached as an activation and as a function."""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class TanhTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_tanh_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 1.0])

        result = ts.tanh(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.0)
        self.assertAlmostEqual(result.tolist()[1], math.tanh(1.0))

    def test_tanh_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.tanh(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
