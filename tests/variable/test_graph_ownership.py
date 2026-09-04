import unittest

import tensors as ts
from tensors.graph import Computation, OperationNode, VariableNode
from tensors.ops import Operation
from tensors.graph.state import reset_graph_state


class VariableGraphOwnershipTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_leaf_variable_owns_a_variable_node(self):
        variable = ts.Variable([1.0], name="weight")

        self.assertEqual(variable.name, "weight")
        self.assertIsInstance(variable.node, VariableNode)
        self.assertEqual(variable.node.label, "var")
        self.assertIs(variable.node.variable, variable)
        self.assertEqual(variable.node.inputs, [])

    def test_operation_result_owns_its_own_variable_node(self):
        variable = ts.Variable([2.0])

        result = variable + 1.0

        self.assertIsInstance(result.node, VariableNode)
        self.assertIs(result.node.variable, result)
        self.assertIsNot(result.node, variable.node)

    def test_variable_node_relationship_holds_for_every_operand_kind(self):
        leaf = ts.Variable([2.0])
        from_tensor = leaf + ts.Tensor([1.0])
        from_scalar = leaf + 1.0

        operands = [
            leaf,
            from_tensor,
            from_scalar,
            *from_tensor.node.producer.operands,
            *from_scalar.node.producer.operands,
        ]
        for variable in operands:
            with self.subTest(variable=variable.name):
                self.assertIsInstance(variable.node, VariableNode)
                self.assertIs(variable.node.variable, variable)
                self.assertNotIsInstance(variable.node, OperationNode)

    def test_operation_node_holds_an_operation_and_no_result_attribute(self):
        variable = ts.Variable([2.0])

        result = variable + 1.0
        producer = result.node.producer

        self.assertIsInstance(producer, OperationNode)
        self.assertIsInstance(producer.operation, Operation)
        self.assertFalse(hasattr(producer, "output_var"))
        self.assertFalse(hasattr(producer, "op_cls"))
        self.assertFalse(hasattr(producer, "args"))

    def test_operands_and_result_are_represented_by_edges(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])

        result = left * right
        producer = result.node.producer

        self.assertEqual(
            [edge.source for edge in producer._in_edges],
            [left.node, right.node],
        )
        self.assertEqual(
            [edge.label for edge in producer._in_edges],
            ["input_0", "input_1"],
        )
        self.assertEqual(producer.operands, (left, right))

        outgoing = producer._out_edges
        self.assertEqual(len(outgoing), 1)
        self.assertIs(outgoing[0].target, result.node)
        self.assertEqual(outgoing[0].label, "result")
        self.assertIs(producer.result, result)

    def test_every_ordinary_operation_node_has_one_result(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        output = ts.sum(ts.relu(x * y + 1.0))

        for node in Computation(output).nodes:
            if isinstance(node, OperationNode):
                with self.subTest(operation=node.label):
                    self.assertEqual(len(node._out_edges), 1)
                    self.assertIsInstance(
                        node._out_edges[0].target,
                        VariableNode,
                    )

    def test_computation_visits_variable_dependencies_before_result(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        result = x * y + 1.0

        computation = Computation(result)
        product = result.node.producer.operands[0]
        constant = result.node.producer.operands[1]

        self.assertEqual(
            computation.nodes,
            [
                x.node,
                y.node,
                product.node.producer,
                product.node,
                constant.node,
                result.node.producer,
                result.node,
            ],
        )

    def test_default_variable_name_is_assigned_when_none_is_provided(self):
        variable = ts.Variable([1.0])

        self.assertRegex(variable.name, r"^v[0-9a-f]{4}$")


if __name__ == "__main__":
    unittest.main()
