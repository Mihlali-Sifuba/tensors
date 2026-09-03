import math
import unittest

import tensors as ts
from tensors.ops import Div, Mul, Pow, Slice
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
        self.assertEqual(result.node.producer.label, "div")
        self.assertEqual(
            result.node.producer.inputs,
            [left.node, right.node],
        )
        self.assertTrue(result.requires_grad)

    def test_scalar_operand_becomes_a_graph_variable(self):
        variable = ts.Variable([1.0, 2.0])

        result = variable * 3.0
        producer = result.node.producer

        self.assertEqual(result.data.tolist(), [3.0, 6.0])
        self.assertEqual(producer.label, "mul")
        self.assertIsInstance(producer.operation, Mul)

        # The scalar is a graph operand, not an operation attribute.
        self.assertEqual(len(producer.operands), 2)
        recorded, scalar = producer.operands
        self.assertIs(recorded, variable)
        self.assertEqual(scalar.data.item(), 3.0)
        self.assertEqual(scalar.shape, ())
        self.assertFalse(scalar.requires_grad)
        self.assertEqual(producer.inputs, [variable.node, scalar.node])
        self.assertFalse(hasattr(producer.operation, "scalar"))

    def test_scalar_operands_preserve_dtype_promotion(self):
        integers = ts.Variable(
            ts.Tensor([1, 2], dtype=ts.int32),
            requires_grad=False,
        )

        promoted = integers * 3
        widened = integers * 3.5

        self.assertIs(promoted.dtype, ts.int32)
        self.assertIs(widened.dtype, ts.float64)

    def test_reverse_operations_produce_expected_values(self):
        variable = ts.Variable([2.0, 4.0])

        difference = 10.0 - variable
        quotient = 8.0 / variable

        self.assertEqual(difference.data.tolist(), [8.0, 6.0])
        self.assertEqual(quotient.data.tolist(), [4.0, 2.0])

    def test_reverse_expressions_are_represented_by_operand_order(self):
        variable = ts.Variable([2.0, 4.0])

        quotient = 3.0 / variable
        producer = quotient.node.producer

        self.assertIsInstance(producer.operation, Div)
        numerator, denominator = producer.operands
        self.assertEqual(numerator.data.item(), 3.0)
        self.assertIs(denominator, variable)
        self.assertEqual(
            [edge.label for edge in producer._in_edges],
            ["input_0", "input_1"],
        )
        # No reverse flag survives: the edges carry the operand order.
        self.assertFalse(hasattr(producer.operation, "reverse"))

    def test_reverse_power_records_the_base_as_the_first_operand(self):
        exponent = ts.Variable([2.0, 3.0])

        result = 2.0 ** exponent
        producer = result.node.producer

        self.assertIsInstance(producer.operation, Pow)
        base, recorded = producer.operands
        self.assertEqual(base.data.item(), 2.0)
        self.assertIs(recorded, exponent)
        self.assertFalse(producer.operation.differentiate_base)
        self.assertTrue(producer.operation.differentiate_exponent)

    def test_negation_and_slicing_create_operation_nodes(self):
        variable = ts.Variable([1.0, 2.0, 3.0])

        negated = -variable
        sliced = variable[1:]

        self.assertEqual(negated.data.tolist(), [-1.0, -2.0, -3.0])
        self.assertEqual(negated.node.producer.label, "neg")
        self.assertEqual(sliced.data.tolist(), [2.0, 3.0])
        self.assertEqual(sliced.node.producer.label, "slice")
        operation = sliced.node.producer.operation
        self.assertIsInstance(operation, Slice)
        # The slice key is operation configuration, not a graph operand.
        self.assertEqual(operation.key, slice(1, None, None))
        self.assertEqual(sliced.node.producer.operands, (variable,))

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
