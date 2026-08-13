import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class NormTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_norm_returns_euclidean_magnitude(self):
        result = ts.norm([3.0, 4.0])

        self.assertEqual(result.shape, ())
        self.assertEqual(result.item(), 5.0)

    def test_norm_is_available_in_linalg_namespace(self):
        self.assertEqual(ts.linalg.norm([3.0, 4.0]).item(), 5.0)

    def test_norm_promotes_integer_input_to_float64(self):
        value = ts.Tensor([3, 4], dtype=ts.int32)

        self.assertIs(ts.norm(value).dtype, ts.float64)

    def test_norm_uses_all_elements_of_a_tensor(self):
        value = ts.Tensor([[1.0, 2.0], [2.0, 4.0]])

        self.assertEqual(ts.norm(value).item(), 5.0)

    def test_norm_records_and_backpropagates(self):
        value = ts.Variable([3.0, 4.0])
        result = ts.norm(value)

        self.assertEqual(result.node.label, "norm")
        self.assertEqual(result.node.inputs, [value.node])

        ts.backward(result)

        self.assertAlmostEqual(value.grad.tolist()[0], 0.6)
        self.assertAlmostEqual(value.grad.tolist()[1], 0.8)

    def test_norm_uses_zero_subgradient_at_zero(self):
        value = ts.Variable([0.0, 0.0])

        ts.backward(ts.norm(value))

        self.assertEqual(value.grad.tolist(), [0.0, 0.0])

    def test_norm_recomputes_from_current_values(self):
        value = ts.Variable([3.0, 4.0])
        result = ts.norm(value)
        computation = Computation(result)

        value.data = ts.Tensor([5.0, 12.0])

        self.assertEqual(computation.forward().item(), 13.0)

    def test_norm_gradient_can_be_differentiated(self):
        value = ts.Variable([3.0, 4.0])
        first = ts.grad(ts.norm(value), value, create_graph=True)
        second = ts.grad(first, value, grad_outputs=ts.Tensor([1.0, 0.0]))

        self.assertAlmostEqual(second.tolist()[0], 0.128)
        self.assertAlmostEqual(second.tolist()[1], -0.096)


if __name__ == "__main__":
    unittest.main()
