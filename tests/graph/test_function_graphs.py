import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class FunctionGraphTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_constructor_rejects_non_callable_function(self):
        with self.assertRaisesRegex(TypeError, "callable"):
            ts.Graph(42)

    def test_unimplemented_subclass_forward_raises_clear_error(self):
        with self.assertRaisesRegex(NotImplementedError, "must implement"):
            ts.Graph()(ts.Tensor([1.0]))

    def test_function_graph_collects_a_captured_parameter_once(self):
        weight = ts.Variable([2.0])

        @ts.Graph
        def model(x):
            return x * weight + weight

        result = model(ts.Tensor([3.0]))

        self.assertEqual(result.data.tolist(), [8.0])
        self.assertEqual(model.parameters(), [weight])

    def test_variable_input_is_used_as_the_original_variable(self):
        @ts.Graph
        def model(x):
            return x * 2.0

        source = ts.Variable([3.0])
        result = model(source)

        self.assertIs(result.node.inputs[0], source.node)
        self.assertEqual(result.data.tolist(), [6.0])


if __name__ == "__main__":
    unittest.main()
