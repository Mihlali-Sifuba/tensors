import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class NestedGraphTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_child_graph_contributes_to_the_parent_computation(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, x):
                return x * self.weight

        class Parent(ts.Graph):
            def __init__(self):
                super().__init__()
                self.scale = Scale()
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return self.scale(x) + self.bias

        model = Parent()
        result = model(ts.Tensor([3.0]))

        self.assertEqual(result.data.tolist(), [7.0])
        self.assertEqual(
            [node.label for node in model.nodes],
            ["var", "var", "mul", "var", "var", "add", "var"],
        )
        self.assertEqual(
            [node.label for node in model.scale.nodes],
            ["var", "var", "mul", "var"],
        )

    def test_parameters_are_deduplicated_when_a_child_is_referenced_twice(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, x):
                return x * self.weight

        class Parent(ts.Graph):
            def __init__(self):
                super().__init__()
                self.left = Scale()
                self.right = self.left

            def forward(self, x):
                return self.left(x) + self.right(x)

        model = Parent()

        self.assertEqual(model.parameters(), [model.left.weight])
        self.assertEqual(model(ts.Tensor([3.0])).data.tolist(), [12.0])

    def test_nested_graphs_trace_fresh_outputs_on_each_parent_call(self):
        @ts.Graph
        def child(x):
            return x * 2.0

        @ts.Graph
        def parent(x):
            return child(x) + 1.0

        first = parent(ts.Tensor([2.0]))
        second = parent(ts.Tensor([4.0]))

        self.assertIsNot(first, second)
        self.assertEqual(first.data.tolist(), [5.0])
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertIs(parent.computation.output, second)
        self.assertIs(
            child.computation.output,
            second.node.producer.operands[0],
        )

    def test_nested_child_computation_stops_at_its_explicit_input(self):
        @ts.Graph
        def child(x):
            return x * 2.0 + 1.0

        @ts.Graph
        def parent(x):
            upstream = x - 3.0
            return child(upstream)

        parent(ts.Tensor([5.0]))

        # The child stops at its boundary Variable: the parent's ``sub``
        # operation is reachable from the parent but not from the child.
        self.assertEqual(
            [node.label for node in child.nodes],
            ["var", "var", "mul", "var", "var", "add", "var"],
        )
        self.assertEqual(
            [node.label for node in parent.nodes],
            [
                "var", "var", "sub", "var",
                "var", "mul", "var",
                "var", "add", "var",
            ],
        )


if __name__ == "__main__":
    unittest.main()
