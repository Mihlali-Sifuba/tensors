import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class AutogradTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_operation_result_is_owned_by_operation_node(self):
        x = ts.Variable([1.0, 2.0])
        result = x + 2.0

        self.assertEqual(result.node.label, "add")
        self.assertIs(result.node.inputs[0], x.node)

    def test_node_labels_do_not_control_execution(self):
        x = ts.Variable([2.0])
        result = x * 3.0
        x.node.label = "input"
        result.node.label = "product"

        replayed = Computation(result).forward()
        ts.backward(result)

        self.assertEqual(replayed.tolist(), [6.0])
        self.assertEqual(x.grad.tolist(), [3.0])

    def test_large_computation_does_not_depend_on_python_recursion(self):
        value = ts.Variable([0.0])
        result = value
        for _ in range(1500):
            result = result + 1.0

        computation = Computation(result)

        self.assertEqual(len(computation.nodes), 1501)
        self.assertEqual(computation.forward().tolist(), [1500.0])

    def test_grad_validates_requested_inputs(self):
        value = ts.Variable([2.0])

        with self.assertRaisesRegex(ValueError, "at least one"):
            ts.grad(value * 2.0, ())
        with self.assertRaisesRegex(TypeError, "input 0 must be a Variable"):
            ts.grad(value * 2.0, [ts.Tensor([2.0])])

    def test_forward_replays_scalar_and_reduction_operations(self):
        x = ts.Variable([1.0, 2.0, 3.0])
        result = ts.mean(x * 2.0 + 1.0)

        replayed = Computation(result).forward()

        self.assertEqual(replayed.tolist(), [5.0])

    def test_sum_propagates_upstream_gradient(self):
        x = ts.Variable([1.0, 2.0])
        w = ts.Variable([3.0, 4.0])
        loss = ts.sum(x * w) * 3.0

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [9.0, 12.0])
        self.assertEqual(w.grad.tolist(), [3.0, 6.0])

    def test_repeated_backward_does_not_reuse_intermediate_gradients(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.sum(x * x)

        ts.backward(loss)
        first = x.grad.tolist()
        ts.backward(loss)
        second = x.grad.tolist()

        self.assertEqual(first, [2.0, 4.0])
        self.assertEqual(second, first)

    def test_grad_clears_a_stale_gradient_for_a_disconnected_input(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])

        self.assertEqual(ts.grad(y * 2.0, y).tolist(), [2.0])
        self.assertIsNone(ts.grad(x * 3.0, y))
        self.assertIsNone(y.grad)

    def test_backward_rejects_an_invalid_gradient_count(self):
        class BrokenOperation:
            @staticmethod
            def forward(value):
                return value

            @staticmethod
            def backward(gradient, value):
                return []

        value = ts.Variable([2.0])
        output = value._from_operation(
            BrokenOperation.forward(value.data),
            "broken",
            BrokenOperation,
            [value],
        )

        with self.assertRaisesRegex(RuntimeError, "returned 0 gradients for 1 inputs"):
            ts.backward(output)

    def test_forward_refreshes_intermediates_used_by_backward(self):
        x = ts.Variable([2.0])
        square = x * x
        fourth_power = square * square
        x.data = ts.Tensor([3.0])

        replayed = Computation(fourth_power).forward()
        ts.backward(fourth_power)

        self.assertEqual(replayed.tolist(), [81.0])
        self.assertEqual(x.grad.tolist(), [108.0])

    def test_slice_scatter_backward(self):
        x = ts.Variable([1.0, 2.0, 3.0])
        loss = ts.sum(x[::-1] * 2.0)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [2.0, 2.0, 2.0])

    def test_dot_backward_for_2d_tensors(self):
        x = ts.Variable([[1.0, 2.0]])
        w = ts.Variable([[3.0], [4.0]])
        result = ts.linalg.dot(x, w)

        replayed = Computation(result).forward()
        ts.backward(result)

        self.assertEqual(replayed.tolist(), [11.0])
        self.assertEqual(x.grad.tolist(), [3.0, 4.0])
        self.assertEqual(w.grad.tolist(), [1.0, 2.0])

    def test_reverse_division_backward(self):
        x = ts.Variable([2.0, 4.0])
        loss = ts.sum(8.0 / x)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [-2.0, -0.5])

    def test_integer_variables_cannot_require_gradients(self):
        with self.assertRaisesRegex(ValueError, "floating-point"):
            ts.Variable(ts.Tensor([1, 2], dtype=ts.int32))

    def test_empty_mean_has_an_empty_gradient(self):
        x = ts.Variable([])
        loss = ts.mean(x)

        ts.backward(loss)

        self.assertEqual(x.grad.shape, (0,))
        self.assertEqual(x.grad.tolist(), [])

    def test_math_namespace_keeps_variable_reductions_differentiable(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.math.sum(x * 3.0)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [3.0, 3.0])

    def test_sgd_updates_external_model_parameters(self):
        weight = ts.Variable([1.0])
        loss = ts.math.sum(weight * 2.0)
        optimizer = ts.optim.SGD([weight], learning_rate=0.1)

        ts.backward(loss)
        optimizer.step()

        self.assertEqual(weight.data.tolist(), [0.8])
        optimizer.zero_grad()
        self.assertIsNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
