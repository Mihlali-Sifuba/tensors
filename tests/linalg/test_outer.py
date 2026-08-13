import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class OuterTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_outer_returns_matrix_of_pairwise_products(self):
        result = ts.outer([1.0, 2.0], [3.0, 4.0, 5.0])

        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.tolist(), [3.0, 4.0, 5.0, 6.0, 8.0, 10.0])

    def test_outer_is_available_in_linalg_namespace(self):
        result = ts.linalg.outer([1.0, 2.0], [3.0, 4.0])

        self.assertEqual(result.tolist(), [3.0, 4.0, 6.0, 8.0])

    def test_outer_promotes_operand_dtypes(self):
        left = ts.Tensor([1.0, 2.0], dtype=ts.float32)
        right = ts.Tensor([3.0, 4.0], dtype=ts.float64)

        self.assertIs(ts.outer(left, right).dtype, ts.float64)

    def test_outer_rejects_non_vector_inputs(self):
        with self.assertRaisesRegex(ValueError, "two 1D vectors"):
            ts.outer([[1.0, 2.0]], [3.0, 4.0])

    def test_outer_records_inputs_and_backpropagates(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0, 5.0])
        result = ts.outer(left, right)

        self.assertEqual(result.node.label, "outer")
        self.assertEqual(result.node.inputs, [left.node, right.node])

        ts.backward(ts.sum(result))

        self.assertEqual(left.grad.tolist(), [12.0, 12.0])
        self.assertEqual(right.grad.tolist(), [3.0, 3.0, 3.0])

    def test_outer_recomputes_from_current_vector_values(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])
        result = ts.outer(left, right)
        computation = Computation(result)

        left.data = ts.Tensor([2.0, 3.0])
        right.data = ts.Tensor([5.0, 7.0])

        self.assertEqual(computation.forward().tolist(), [10.0, 14.0, 15.0, 21.0])

    def test_outer_gradient_can_be_differentiated(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0, 5.0])
        loss = ts.sum(ts.outer(left, right))

        left_gradient = ts.grad(loss, left, create_graph=True)
        mixed_gradient = ts.grad(ts.sum(left_gradient), right)

        self.assertEqual(mixed_gradient.tolist(), [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
