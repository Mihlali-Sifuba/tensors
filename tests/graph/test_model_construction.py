import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.computation.compiler import Compiler
from tensors.graph.node import VariableNode
from tensors.graph.state import reset_graph_state
from tensors.ops import Add, Mul


class Linear(ts.Graph):
    """The model shape this lifecycle is built for."""

    def __init__(self):
        super().__init__()
        self.w = ts.Variable([[2.0]], name="w")
        self.b = ts.Variable([1.0], name="b")

    def forward(self, x):
        return x @ self.w + self.b


class ModelConstructionTests(unittest.TestCase):
    """A subclass builds and compiles its graph as it is constructed."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_construction_builds_the_structural_graph(self):
        model = Linear()

        structure = model._structure
        self.assertIsNotNone(structure)
        self.assertEqual(len(structure.inputs), 1)
        self.assertIsInstance(structure.inputs[0], VariableNode)
        # The graph exists before any value does.
        self.assertFalse(structure.inputs[0].is_bound)
        self.assertEqual(
            [node.label for node in model.nodes],
            ["var", "var", "dot", "var", "var", "add", "var"],
        )

    def test_parameters_take_part_as_their_own_vertices(self):
        model = Linear()

        product = model._structure.computations[0]._variable_nodes
        self.assertIn(model.w.node, product)
        self.assertIn(model.b.node, product)
        # The build used the parameters that __init__ created, not copies.
        self.assertIs(model.w.node.variable, model.w)
        self.assertIs(model.b.node.variable, model.b)

    def test_construction_executes_no_kernels(self):
        executed = []
        originals = {Add: Add.forward, Mul: Mul.forward}

        def counted(self, *args):
            executed.append(type(self).__name__)
            return originals[type(self)](self, *args)

        with patch.object(Add, "forward", counted):
            model = Linear()

        self.assertEqual(executed, [])
        self.assertFalse(model._structure.inputs[0].is_bound)

    def test_the_model_is_compiled_once_during_construction(self):
        compilations = []
        original = Compiler.compile

        def counted(self):
            compilations.append(self)
            return original(self)

        with patch.object(Compiler, "compile", counted):
            model = Linear()

        self.assertEqual(len(compilations), 1)
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in compilations[0].instructions
            ],
            ["dot", "add"],
        )
        self.assertIs(
            model._structure.computations[0]._instructions,
            compilations[0].instructions,
        )


class ModelExecutionTests(unittest.TestCase):
    """Calling a built model binds its inputs and replays its program."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_call_executes_the_prebuilt_computation(self):
        model = Linear()
        computation = model._structure.computations[0]

        result = model(ts.Tensor([[3.0]]))

        self.assertIs(model.computation, computation)
        self.assertIs(computation.output, result)
        self.assertEqual(result.data.tolist(), [7.0])
        # The call bound the input vertex the graph was built with.
        self.assertTrue(model._structure.inputs[0].is_bound)
        self.assertEqual(
            model._structure.inputs[0].variable.data.tolist(), [3.0]
        )

    def test_a_call_does_not_re_execute_the_python_forward(self):
        calls = []

        class Counted(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0]], name="w")

            def forward(self, x):
                calls.append(x)
                return x @ self.w

        model = Counted()
        self.assertEqual(len(calls), 1)

        model(ts.Tensor([[3.0]]))
        model(ts.Tensor([[4.0]]))

        # Only the structural build ran forward; the calls replayed it.
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], VariableNode)

    def test_repeated_calls_update_the_input_and_replay(self):
        model = Linear()
        structure = model._structure

        first = model(ts.Tensor([[3.0]]))
        second = model(ts.Tensor([[4.0]]))

        self.assertIs(first, second)
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertIs(model._structure, structure)
        self.assertIs(model._structure.computations[0], structure.computations[0])

    def test_structural_identity_is_stable_across_calls(self):
        model = Linear()
        nodes = list(model.nodes)
        output_node = model._structure.computations[0]._variable_nodes[-1]

        model(ts.Tensor([[3.0]]))
        model(ts.Tensor([[5.0]]))

        self.assertEqual(list(model.nodes), nodes)
        self.assertIs(
            model._structure.computations[0]._variable_nodes[-1], output_node
        )
        self.assertIs(output_node.variable, model.computation.output)

    def test_outputs_are_numerically_correct(self):
        model = Linear()

        self.assertEqual(model(ts.Tensor([[3.0]])).data.tolist(), [7.0])
        self.assertEqual(model(ts.Tensor([[0.5]])).data.tolist(), [2.0])
        self.assertEqual(model(ts.Tensor([[-2.0]])).data.tolist(), [-3.0])

    def test_gradients_still_reach_the_parameters(self):
        model = Linear()

        prediction = model(ts.Tensor([[3.0]]))
        ts.backward(ts.sum(prediction))

        self.assertEqual(model.w.grad.tolist(), [3.0])
        self.assertEqual(model.b.grad.tolist(), [1.0])
        self.assertEqual(model.parameters(), [model.w, model.b])

    def test_a_nested_model_contributes_its_operations(self):
        class Inner(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[3.0]], name="inner_w")

            def forward(self, x):
                return x @ self.w

        class Outer(ts.Graph):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.b = ts.Variable([1.0], name="outer_b")

            def forward(self, x):
                return self.inner(x) + self.b

        model = Outer()

        self.assertEqual(
            [
                instruction.operation.name
                for instruction in model._structure.computations[0]._instructions
            ],
            ["dot", "add"],
        )
        result = model(ts.Tensor([[2.0]]))
        self.assertEqual(result.data.tolist(), [7.0])
        ts.backward(ts.sum(result))
        self.assertEqual(model.inner.w.grad.tolist(), [2.0])
        self.assertEqual(model.b.grad.tolist(), [1.0])
        self.assertEqual(model.parameters(), [model.inner.w, model.b])

    def test_a_functional_graph_keeps_tracing(self):
        weight = ts.Variable([2.0], name="weight")

        @ts.Graph
        def model(x):
            return x * weight + 1.0

        self.assertIsNone(model._structure)
        first = model(ts.Tensor([3.0]))
        second = model(ts.Tensor([4.0]))

        # The functional form still records a fresh computation per call.
        self.assertIsNot(first, second)
        self.assertEqual(first.data.tolist(), [7.0])
        self.assertEqual(second.data.tolist(), [9.0])

    def test_a_model_needing_runtime_values_keeps_tracing(self):
        class Scaled(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([2.0], name="w")

            def forward(self, x):
                # A Python scalar cannot be recorded before the value beside
                # it exists, so this model keeps the tracing lifecycle.
                return x * self.w + 1.0

        model = Scaled()

        self.assertIsNone(model._structure)
        self.assertEqual(model(ts.Tensor([3.0])).data.tolist(), [7.0])
        self.assertEqual(model(ts.Tensor([4.0])).data.tolist(), [9.0])


if __name__ == "__main__":
    unittest.main()
