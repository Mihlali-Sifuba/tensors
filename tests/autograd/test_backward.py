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

    def test_grad_preserves_a_stale_gradient_for_a_disconnected_input(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        previous = ts.Tensor([7.0])
        y.grad = previous

        self.assertEqual(ts.grad(y * 2.0, y).tolist(), [2.0])
        self.assertIsNone(ts.grad(x * 3.0, y))
        self.assertIs(y.grad, previous)

    def test_grad_does_not_modify_reachable_grad_attributes(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        product = x * y
        output = ts.sum(product)
        previous = {
            x: ts.Tensor([10.0]),
            y: ts.Tensor([20.0]),
            product: ts.Tensor([30.0]),
            output: ts.Tensor([40.0]),
        }
        for variable, gradient in previous.items():
            variable.grad = gradient

        x_gradient, y_gradient = ts.grad(output, (x, y))

        self.assertEqual(x_gradient.tolist(), [3.0])
        self.assertEqual(y_gradient.tolist(), [2.0])
        for variable, gradient in previous.items():
            self.assertIs(variable.grad, gradient)

    def test_create_graph_grad_does_not_modify_grad_attributes(self):
        x = ts.Variable([2.0])
        output = x ** 3.0
        previous_x_gradient = ts.Tensor([7.0])
        previous_output_gradient = ts.Tensor([8.0])
        x.grad = previous_x_gradient
        output.grad = previous_output_gradient

        result = ts.grad(output, x, create_graph=True)

        self.assertEqual(result.data.tolist(), [12.0])
        self.assertIs(x.grad, previous_x_gradient)
        self.assertIs(output.grad, previous_output_gradient)

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

    def test_failed_backward_does_not_partially_replace_gradients(self):
        class BrokenOperation:
            @staticmethod
            def forward(value):
                return value

            @staticmethod
            def backward(gradient, value):
                raise RuntimeError("deliberate failure")

        value = ts.Variable([2.0])
        output = value._from_operation(
            BrokenOperation.forward(value.data),
            "broken",
            BrokenOperation,
            [value],
        )
        previous_value_gradient = ts.Tensor([7.0])
        previous_output_gradient = ts.Tensor([8.0])
        value.grad = previous_value_gradient
        output.grad = previous_output_gradient

        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            ts.backward(output)

        self.assertIs(value.grad, previous_value_gradient)
        self.assertIs(output.grad, previous_output_gradient)

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
