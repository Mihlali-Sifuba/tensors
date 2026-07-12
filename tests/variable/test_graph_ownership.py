import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class VariableGraphOwnershipTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_leaf_variable_owns_a_variable_node(self):
        variable = ts.Variable([1.0], name="weight")

        self.assertEqual(variable.name, "weight")
        self.assertEqual(variable.node.label, "var")
        self.assertIs(variable.node.output_var, variable)
        self.assertEqual(variable.node.inputs, [])

    def test_operation_variable_is_owned_by_its_operation_node(self):
        variable = ts.Variable([2.0])

        result = variable + 1.0

        self.assertIs(result.node.output_var, result)
        self.assertIsNot(result.node, variable.node)
        self.assertEqual(result.node.inputs, [variable.node])

    def test_computation_visits_variable_dependencies_before_result(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        result = x * y + 1.0

        computation = Computation(result)

        self.assertEqual(
            computation.nodes,
            [x.node, y.node, result.node.inputs[0], result.node],
        )

    def test_default_variable_name_is_assigned_when_none_is_provided(self):
        variable = ts.Variable([1.0])

        self.assertRegex(variable.name, r"^v[0-9a-f]{4}$")


if __name__ == "__main__":
    unittest.main()
