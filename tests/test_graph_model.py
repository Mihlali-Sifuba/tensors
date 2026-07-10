import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class GraphModelTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_graph_package_owns_structural_types(self):
        self.assertEqual(ts.graph.Edge.__module__, "tensors.graph.edge")
        self.assertEqual(ts.graph.Node.__module__, "tensors.graph.node")

    def test_subclass_traces_then_replays_with_tensor_inputs(self):
        class Linear(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        model = Linear()
        first = model(ts.Tensor([3.0]))
        self.assertEqual(first.data.tolist(), [7.0])
        second = model(ts.Tensor([4.0]))

        self.assertIs(first, second)
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertEqual(len(model.nodes), 5)
        self.assertEqual(model.parameters(), [model.weight, model.bias])

    def test_function_graph_works_as_a_decorator_target(self):
        weight = ts.Variable([3.0])

        @ts.Graph
        def model(x):
            return x * weight

        result = model(ts.Tensor([2.0]))

        self.assertEqual(result.data.tolist(), [6.0])
        self.assertEqual(model.parameters(), [weight])

    def test_external_loss_backpropagates_into_model_parameters(self):
        class Linear(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        model = Linear()
        loss = ts.math.sum(model(ts.Tensor([3.0])))

        ts.backward(loss)

        self.assertEqual(model.weight.grad.tolist(), [3.0])
        self.assertEqual(model.bias.grad.tolist(), [1.0])

    def test_replay_rejects_an_input_shape_change(self):
        @ts.Graph
        def model(x):
            return x * 2.0

        model(ts.Tensor([1.0]))

        with self.assertRaisesRegex(ValueError, "Call rebuild"):
            model(ts.Tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
