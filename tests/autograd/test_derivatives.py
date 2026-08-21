import unittest

import tensors as ts
from tensors.graph.state import get_graph_state, reset_graph_state


class DerivativeMatrixTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_jacobian_contains_every_output_input_derivative(self):
        value = ts.Variable([2.0, 3.0])
        output = ts.concat([value[0] ** 2.0, value[0] * value[1]])

        result = ts.jacobian(output, value)

        self.assertIsInstance(result, ts.Tensor)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [4.0, 0.0, 3.0, 2.0])

    def test_jacobian_shape_is_output_shape_plus_input_shape(self):
        value = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
        output = ts.sum(value, axis=1)

        result = ts.jacobian(output, value)

        self.assertEqual(result.shape, (2, 2, 2))
        self.assertEqual(
            result.tolist(),
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        )

    def test_scalar_shaped_output_does_not_add_a_jacobian_axis(self):
        value = ts.Variable([2.0, 3.0])
        output = ts.reshape(ts.sum(value ** 2.0), ())

        result = ts.jacobian(output, value)

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [4.0, 6.0])

    def test_jacobian_returns_one_result_per_input(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = ts.concat([left * right, left + right])

        left_result, right_result = ts.jacobian(output, (left, right))

        self.assertEqual(left_result.shape, (2, 1))
        self.assertEqual(right_result.shape, (2, 1))
        self.assertEqual(left_result.tolist(), [3.0, 1.0])
        self.assertEqual(right_result.tolist(), [2.0, 1.0])

    def test_jacobian_uses_zero_for_a_disconnected_input(self):
        connected = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
        disconnected = ts.Variable(
            ts.Tensor([3.0, 4.0], dtype=ts.float32)
        )
        output = connected ** 2.0

        result = ts.jacobian(output, disconnected)

        self.assertEqual(result.shape, (1, 2))
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [0.0, 0.0])

    def test_jacobian_preserves_existing_grad_attributes(self):
        value = ts.Variable([2.0, 3.0])
        output = value ** 2.0
        old_input_gradient = ts.Tensor([7.0, 8.0])
        old_output_gradient = ts.Tensor([9.0, 10.0])
        value.grad = old_input_gradient
        output.grad = old_output_gradient

        ts.jacobian(output, value)

        self.assertIs(value.grad, old_input_gradient)
        self.assertIs(output.grad, old_output_gradient)

    def test_create_graph_keeps_jacobian_differentiable(self):
        value = ts.Variable([2.0])
        output = value ** 3.0

        result = ts.jacobian(output, value, create_graph=True)
        second = ts.grad(ts.sum(result), value)

        self.assertIsInstance(result, ts.Variable)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.data.tolist(), [12.0])
        self.assertEqual(second.tolist(), [12.0])

    def test_hessian_contains_every_second_derivative(self):
        value = ts.Variable([2.0, 3.0])
        output = ts.sum(value[0] ** 2.0 + value[0] * value[1])

        result = ts.hessian(output, value)

        self.assertIsInstance(result, ts.Tensor)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [2.0, 1.0, 1.0, 0.0])

    def test_tensor_hessian_releases_its_temporary_derivative_graph(self):
        value = ts.Variable([2.0, 3.0])
        output = ts.sum(value ** 3.0)
        state = get_graph_state()
        original_nodes = list(state.nodes)
        original_edges = list(state.edges)
        original_outputs = list(value.node._out_edges)

        ts.hessian(output, value)

        self.assertIs(get_graph_state(), state)
        self.assertEqual(state.nodes, original_nodes)
        self.assertEqual(state.edges, original_edges)
        self.assertEqual(value.node._out_edges, original_outputs)

    def test_hessian_returns_blocks_for_multiple_inputs(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = ts.sum((left ** 2.0) * right + right ** 3.0)

        result = ts.hessian(output, (left, right))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0].tolist(), [6.0])
        self.assertEqual(result[0][1].tolist(), [4.0])
        self.assertEqual(result[1][0].tolist(), [4.0])
        self.assertEqual(result[1][1].tolist(), [18.0])
        for row in result:
            for block in row:
                self.assertEqual(block.shape, (1, 1))

    def test_hessian_uses_zero_blocks_for_disconnected_inputs(self):
        connected = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
        disconnected = ts.Variable(
            ts.Tensor([3.0, 4.0], dtype=ts.float32)
        )
        output = ts.sum(connected ** 2.0)

        result = ts.hessian(output, (connected, disconnected))

        self.assertEqual(result[0][0].tolist(), [2.0])
        self.assertEqual(result[0][1].tolist(), [0.0, 0.0])
        self.assertEqual(result[1][0].tolist(), [0.0, 0.0])
        self.assertEqual(result[1][1].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertIs(result[0][1].dtype, ts.float32)
        self.assertIs(result[1][0].dtype, ts.float64)
        self.assertIs(result[1][1].dtype, ts.float32)

    def test_create_graph_keeps_hessian_differentiable(self):
        value = ts.Variable([2.0])
        output = ts.sum(value ** 3.0)

        result = ts.hessian(output, value, create_graph=True)
        third = ts.grad(ts.sum(result), value)

        self.assertIsInstance(result, ts.Variable)
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.data.tolist(), [12.0])
        self.assertEqual(third.tolist(), [6.0])

    def test_hessian_requires_a_single_element_output(self):
        value = ts.Variable([2.0, 3.0])

        with self.assertRaisesRegex(ValueError, "exactly one element"):
            ts.hessian(value ** 2.0, value)

    def test_derivative_matrices_validate_outputs_and_inputs(self):
        value = ts.Variable([2.0])
        frozen = ts.Variable([3.0], requires_grad=False)

        with self.assertRaisesRegex(TypeError, "output must be a Variable"):
            ts.jacobian(ts.Tensor([2.0]), value)
        with self.assertRaisesRegex(TypeError, "input 0 must be a Variable"):
            ts.jacobian(value ** 2.0, [ts.Tensor([2.0])])
        with self.assertRaisesRegex(ValueError, "requires_grad=True"):
            ts.hessian(value ** 2.0, frozen)


if __name__ == "__main__":
    unittest.main()
