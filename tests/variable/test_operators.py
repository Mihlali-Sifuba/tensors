import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariableOperatorTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_binary_variable_operations_preserve_input_order(self):
        left = ts.Variable([8.0, 12.0])
        right = ts.Variable([2.0, 3.0])

        result = left / right

        self.assertEqual(result.data.tolist(), [4.0, 4.0])
        self.assertEqual(result.node.label, "div")
        self.assertEqual(result.node.inputs, [left.node, right.node])
        self.assertTrue(result.requires_grad)

    def test_scalar_operation_records_the_scalar_argument(self):
        variable = ts.Variable([1.0, 2.0])

        result = variable * 3.0

        self.assertEqual(result.data.tolist(), [3.0, 6.0])
        self.assertEqual(result.node.label, "mul")
        self.assertEqual(result.node.args, {"scalar": 3.0})
        self.assertEqual(result.node.inputs, [variable.node])

    def test_reverse_operations_produce_expected_values(self):
        variable = ts.Variable([2.0, 4.0])

        difference = 10.0 - variable
        quotient = 8.0 / variable

        self.assertEqual(difference.data.tolist(), [8.0, 6.0])
        self.assertEqual(quotient.data.tolist(), [4.0, 2.0])
        self.assertEqual(quotient.node.args, {"scalar": 8.0, "reverse": True})

    def test_negation_and_slicing_create_operation_nodes(self):
        variable = ts.Variable([1.0, 2.0, 3.0])

        negated = -variable
        sliced = variable[1:]

        self.assertEqual(negated.data.tolist(), [-1.0, -2.0, -3.0])
        self.assertEqual(negated.node.label, "neg")
        self.assertEqual(sliced.data.tolist(), [2.0, 3.0])
        self.assertEqual(sliced.node.label, "slice")
        self.assertEqual(sliced.node.args, {"key": slice(1, None, None)})

    def test_requires_grad_is_inferred_from_all_operation_inputs(self):
        frozen = ts.Variable([2.0], requires_grad=False)
        trainable = ts.Variable([3.0])

        frozen_only = frozen + 1.0
        mixed = frozen * trainable

        self.assertFalse(frozen_only.requires_grad)
        self.assertTrue(mixed.requires_grad)

    def test_tensor_left_operations_build_variable_graphs(self):
        constant = ts.Tensor([3.0])
        variable = ts.Variable([2.0])
        cases = [
            ("add", lambda: constant + variable, 5.0, 1.0),
            ("subtract", lambda: constant - variable, 1.0, -1.0),
            ("multiply", lambda: constant * variable, 6.0, 3.0),
            ("divide", lambda: constant / variable, 1.5, -0.75),
            ("power", lambda: constant ** variable, 9.0, 9.0 * math.log(3.0)),
        ]

        for name, operation, expected_value, expected_gradient in cases:
            with self.subTest(operation=name):
                result = operation()

                self.assertIsInstance(result, ts.Variable)
                self.assertAlmostEqual(result.data.item(), expected_value)
                gradient = ts.grad(ts.sum(result), variable)
                self.assertAlmostEqual(gradient.item(), expected_gradient)


if __name__ == "__main__":
    unittest.main()
