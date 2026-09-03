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

    def test_norm_is_axis_aware_and_supports_keepdims(self):
        value = ts.Tensor([[3.0, 4.0], [5.0, 12.0]])

        rows = ts.norm(value, axis=1)
        columns = ts.norm(value, axis=0, keepdims=True)

        self.assertEqual(rows.shape, (2,))
        self.assertEqual(rows.tolist(), [5.0, 13.0])
        self.assertEqual(columns.shape, (1, 2))
        self.assertAlmostEqual(columns.tolist()[0], 34 ** 0.5)
        self.assertAlmostEqual(columns.tolist()[1], 160 ** 0.5)

    def test_norm_avoids_overflow_and_underflow(self):
        value = ts.Tensor([[1.0e308, 0.0], [1.0e-300, 0.0]])

        result = ts.norm(value, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [1.0e308, 1.0e-300])

    def test_norm_accepts_multiple_axes(self):
        value = ts.Tensor(
            [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 0.0],
            shape=(2, 2, 2),
        )

        result = ts.norm(value, axis=(1, 2))

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [5.0, 13.0])

    def test_norm_records_and_backpropagates(self):
        value = ts.Variable([3.0, 4.0])
        result = ts.norm(value)

        self.assertEqual(result.node.producer.label, "norm")
        self.assertEqual(result.node.producer.inputs, [value.node])

        ts.backward(result)

        self.assertAlmostEqual(value.grad.tolist()[0], 0.6)
        self.assertAlmostEqual(value.grad.tolist()[1], 0.8)

    def test_norm_uses_zero_subgradient_at_zero(self):
        value = ts.Variable([0.0, 0.0])

        ts.backward(ts.norm(value))

        self.assertEqual(value.grad.tolist(), [0.0, 0.0])

    def test_axis_norm_backpropagates_per_reduction_group(self):
        value = ts.Variable([[3.0, 4.0], [5.0, 12.0]])

        ts.backward(ts.sum(ts.norm(value, axis=1)))

        gradient = value.grad.tolist()
        self.assertAlmostEqual(gradient[0], 0.6)
        self.assertAlmostEqual(gradient[1], 0.8)
        self.assertAlmostEqual(gradient[2], 5.0 / 13.0)
        self.assertAlmostEqual(gradient[3], 12.0 / 13.0)

    def test_extreme_norm_values_have_finite_gradients(self):
        value = ts.Variable([[1.0e308, 0.0], [1.0e-300, 0.0]])

        ts.backward(ts.sum(ts.norm(value, axis=1)))

        self.assertEqual(value.grad.tolist(), [1.0, 0.0, 1.0, 0.0])

    def test_axis_norm_recomputes_with_recorded_axis(self):
        value = ts.Variable([[3.0, 4.0], [5.0, 12.0]])
        result = ts.norm(value, axis=1, keepdims=True)
        computation = Computation(result)

        value.data = ts.Tensor([[6.0, 8.0], [8.0, 15.0]])

        recomputed = computation.forward()
        self.assertEqual(recomputed.shape, (2, 1))
        self.assertEqual(recomputed.tolist(), [10.0, 17.0])

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

    def test_large_norm_gradient_can_be_differentiated(self):
        value = ts.Variable([1.0e308, 0.0])
        first = ts.grad(ts.norm(value), value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([0.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [1.0, 0.0])
        self.assertEqual(second.tolist(), [0.0, 1.0e-308])

    def test_axis_norm_gradient_can_be_differentiated(self):
        value = ts.Variable([[3.0, 4.0]])
        first = ts.grad(ts.norm(value, axis=1), value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([[1.0, 0.0]]),
        )

        self.assertEqual(first.shape, (1, 2))
        self.assertAlmostEqual(second.tolist()[0], 0.128)
        self.assertAlmostEqual(second.tolist()[1], -0.096)


if __name__ == "__main__":
    unittest.main()
