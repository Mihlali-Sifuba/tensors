import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class GraphOutputTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_nested_output_containers_create_one_computation_per_variable(self):
        @ts.Graph
        def model(x):
            return [x + 1.0, (x * 2.0, [x - 1.0])]

        outputs = model(ts.Tensor([3.0]))

        self.assertEqual(outputs[0].data.tolist(), [4.0])
        self.assertEqual(outputs[1][0].data.tolist(), [6.0])
        self.assertEqual(outputs[1][1][0].data.tolist(), [2.0])
        self.assertEqual(len(model.computations), 3)

    def test_graph_captures_unique_nodes_and_edges_for_shared_input(self):
        @ts.Graph
        def model(x):
            return x + 1.0, x * 2.0

        model(ts.Tensor([3.0]))

        self.assertEqual(
            [node.label for node in model.nodes],
            ["var", "var", "add", "var", "var", "mul", "var"],
        )
        self.assertEqual(
            [edge.label for edge in model.edges],
            ["input_0", "input_1", "result", "input_0", "input_1", "result"],
        )

    def test_nodes_and_edges_properties_return_independent_lists(self):
        @ts.Graph
        def model(x):
            return x + 1.0

        model(ts.Tensor([3.0]))
        nodes = model.nodes
        edges = model.edges
        nodes.clear()
        edges.clear()

        self.assertEqual(len(model.nodes), 4)
        self.assertEqual(len(model.edges), 3)

    def test_computation_is_unavailable_until_the_graph_has_been_called(self):
        @ts.Graph
        def model(x):
            return x + 1.0

        with self.assertRaisesRegex(RuntimeError, "has not been called"):
            _ = model.computation


if __name__ == "__main__":
    unittest.main()
