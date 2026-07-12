import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariableSoftmaxTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_softmax_keeps_variable_history_and_applies_its_jacobian(self):
        logits = ts.Variable([1.0, 2.0, 3.0])
        probabilities = ts.softmax(logits)
        upstream = ts.Tensor([1.0, 2.0, 3.0])

        ts.backward(probabilities, upstream)

        weighted_sum = sum(
            gradient * probability
            for gradient, probability in zip(upstream.tolist(), probabilities.data.tolist())
        )
        expected = [
            probability * (gradient - weighted_sum)
            for gradient, probability in zip(upstream.tolist(), probabilities.data.tolist())
        ]
        self.assertEqual(probabilities.node.label, "softmax")
        for actual, expected_value in zip(logits.grad.tolist(), expected):
            self.assertAlmostEqual(actual, expected_value)


if __name__ == "__main__":
    unittest.main()
