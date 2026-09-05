import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariableConcatTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_concat_returns_a_differentiable_variable(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0])
        result = ts.concat([left, right])

        ts.backward(ts.sum(result))

        self.assertEqual(result.node.producer.label, "concat")
        self.assertEqual(result.data.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(left.grad.tolist(), [1.0, 1.0])
        self.assertEqual(right.grad.tolist(), [1.0])

    def test_concat_backward_splits_gradients_along_a_nonzero_axis(self):
        left = ts.Variable([[1.0], [2.0]])
        right = ts.Variable([[3.0, 4.0], [5.0, 6.0]])
        result = ts.concat([left, right], axis=1)

        ts.backward(result, ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        self.assertEqual(left.grad.shape, (2, 1))
        self.assertEqual(left.grad.tolist(), [1.0, 4.0])
        self.assertEqual(right.grad.shape, (2, 2))
        self.assertEqual(right.grad.tolist(), [2.0, 3.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
