import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class VariableGradientTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_addition_and_subtraction_have_expected_gradients(self):
        left = ts.Variable([2.0, 3.0])
        right = ts.Variable([5.0, 7.0])
        loss = ts.sum(left - right + left)

        ts.backward(loss)

        self.assertEqual(left.grad.tolist(), [2.0, 2.0])
        self.assertEqual(right.grad.tolist(), [-1.0, -1.0])

    def test_scalar_multiplication_scales_the_gradient(self):
        variable = ts.Variable([1.0, 2.0, 3.0])
        loss = ts.sum(variable * 2.5)

        ts.backward(loss)

        self.assertEqual(variable.grad.tolist(), [2.5, 2.5, 2.5])

    def test_shared_variable_accumulates_gradients_from_both_branches(self):
        variable = ts.Variable([2.0, 3.0])
        loss = ts.sum(variable * variable + variable)

        ts.backward(loss)

        self.assertEqual(variable.grad.tolist(), [5.0, 7.0])

    def test_slice_backward_only_updates_selected_positions(self):
        variable = ts.Variable([1.0, 2.0, 3.0, 4.0])
        loss = ts.sum(variable[1:3])

        ts.backward(loss)

        self.assertEqual(variable.grad.tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_computation_accepts_an_explicit_output_gradient(self):
        variable = ts.Variable([2.0, 4.0])
        result = variable * 3.0

        Computation(result).backward(ts.Tensor([0.5, 2.0]))

        self.assertEqual(variable.grad.tolist(), [1.5, 6.0])

    def test_backward_replaces_a_previous_leaf_gradient(self):
        variable = ts.Variable([2.0])
        loss = ts.sum(variable * 4.0)

        ts.backward(loss)
        ts.backward(loss)

        self.assertEqual(variable.grad.tolist(), [4.0])

    def test_broadcast_addition_reduces_gradients_to_input_shapes(self):
        column = ts.Variable([[1.0], [2.0]])
        matrix = ts.Variable([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        loss = ts.sum(column + matrix)

        ts.backward(loss)

        self.assertEqual(column.grad.shape, (2, 1))
        self.assertEqual(column.grad.tolist(), [3.0, 3.0])
        self.assertEqual(matrix.grad.shape, (2, 3))
        self.assertEqual(matrix.grad.tolist(), [1.0] * 6)

    def test_broadcast_multiplication_reduces_gradients_to_input_shapes(self):
        column = ts.Variable([[1.0], [2.0]])
        matrix = ts.Variable([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
        loss = ts.sum(column * matrix)

        ts.backward(loss)

        self.assertEqual(column.grad.shape, (2, 1))
        self.assertEqual(column.grad.tolist(), [12.0, 21.0])
        self.assertEqual(matrix.grad.shape, (2, 3))
        self.assertEqual(matrix.grad.tolist(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_broadcast_division_reduces_gradients_to_input_shapes(self):
        column = ts.Variable([[8.0], [12.0]])
        matrix = ts.Variable([[2.0, 4.0, 8.0], [3.0, 6.0, 12.0]])
        loss = ts.sum(column / matrix)

        ts.backward(loss)

        self.assertEqual(column.grad.shape, (2, 1))
        self.assertAlmostEqual(column.grad.tolist()[0], 0.875)
        self.assertAlmostEqual(column.grad.tolist()[1], 0.5833333333333333)
        self.assertEqual(matrix.grad.shape, (2, 3))
        expected = [-2.0, -0.5, -0.125, -4.0 / 3.0, -1.0 / 3.0, -1.0 / 12.0]
        for actual, expected_value in zip(matrix.grad.tolist(), expected):
            self.assertAlmostEqual(actual, expected_value)


if __name__ == "__main__":
    unittest.main()
