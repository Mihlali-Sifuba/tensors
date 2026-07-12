import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariablePowerTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_scalar_exponent_backpropagates_to_the_base(self):
        base = ts.Variable([2.0, 3.0])
        loss = ts.sum(base ** 3.0)

        ts.backward(loss)

        self.assertEqual(loss.data.tolist(), [35.0])
        self.assertEqual(base.grad.tolist(), [12.0, 27.0])

    def test_tensor_exponent_backpropagates_to_both_inputs(self):
        base = ts.Variable([2.0, 3.0])
        exponent = ts.Variable([2.0, 2.0])
        loss = ts.sum(base ** exponent)

        ts.backward(loss)

        self.assertEqual(base.grad.tolist(), [4.0, 6.0])
        self.assertAlmostEqual(exponent.grad.tolist()[0], 4.0 * math.log(2.0))
        self.assertAlmostEqual(exponent.grad.tolist()[1], 9.0 * math.log(3.0))

    def test_broadcast_power_reduces_the_base_gradient_to_its_shape(self):
        base = ts.Variable([[2.0], [3.0]])
        exponent = ts.Variable([[1.0, 2.0, 3.0], [2.0, 1.0, 2.0]])
        loss = ts.sum(base ** exponent)

        ts.backward(loss)

        self.assertEqual(base.grad.shape, (2, 1))
        self.assertEqual(base.grad.tolist(), [17.0, 13.0])
        self.assertEqual(exponent.grad.shape, (2, 3))

    def test_reverse_power_backpropagates_to_the_exponent(self):
        exponent = ts.Variable([2.0, 3.0])
        loss = ts.sum(2.0 ** exponent)

        ts.backward(loss)

        self.assertAlmostEqual(exponent.grad.tolist()[0], 4.0 * math.log(2.0))
        self.assertAlmostEqual(exponent.grad.tolist()[1], 8.0 * math.log(2.0))


if __name__ == "__main__":
    unittest.main()
