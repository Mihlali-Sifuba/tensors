import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class VariableRecomputationTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_forward_uses_current_leaf_values_for_binary_operations(self):
        inputs = ts.Variable([2.0, 3.0])
        weights = ts.Variable([4.0, 5.0])
        result = inputs * weights + 1.0
        computation = Computation(result)

        inputs.data = ts.Tensor([6.0, 7.0])
        weights.data = ts.Tensor([2.0, 3.0])

        self.assertEqual(computation.forward().tolist(), [13.0, 22.0])
        self.assertEqual(result.data.tolist(), [13.0, 22.0])

    def test_forward_refreshes_a_variable_math_operation(self):
        variable = ts.Variable([1.0, 4.0])
        result = ts.sqrt(variable)
        computation = Computation(result)

        variable.data = ts.Tensor([9.0, 16.0])

        self.assertEqual(computation.forward().tolist(), [3.0, 4.0])

    def test_forward_recomputes_reverse_scalar_division(self):
        variable = ts.Variable([2.0, 4.0])
        result = 12.0 / variable
        computation = Computation(result)

        variable.data = ts.Tensor([3.0, 6.0])

        self.assertEqual(computation.forward().tolist(), [4.0, 2.0])

    def test_forward_keeps_the_recorded_nodes_stable(self):
        variable = ts.Variable([2.0])
        result = ts.exp(variable * 2.0)
        computation = Computation(result)
        original_nodes = computation.nodes

        variable.data = ts.Tensor([3.0])
        computation.forward()

        self.assertEqual(computation.nodes, original_nodes)
        self.assertEqual(result.data.tolist(), [ts.exp(ts.Tensor([6.0])).item()])


if __name__ == "__main__":
    unittest.main()
