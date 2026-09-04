import ast
import inspect
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation
from tensors.graph import graph as graph_module
from tensors.graph.computation import computation as computation_module
from tensors.graph.computation.compiler import Compiler
from tensors.graph.state import reset_graph_state


def runtime_imported_names(module) -> set[str]:
    """Return the imports that execute, ignoring TYPE_CHECKING blocks."""
    tree = ast.parse(inspect.getsource(module))
    guarded = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
        for statement in node.body
        for child in ast.walk(statement)
    }
    names: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


class ComputationLeavesGraphStructureBehindTests(unittest.TestCase):
    """A Computation works in the compiled domain only."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_no_runtime_import_of_graph_structure(self):
        runtime = runtime_imported_names(computation_module)
        for name in ("Node", "VariableNode", "Edge", "..node", "..edge"):
            with self.subTest(name=name):
                self.assertNotIn(name, runtime)
        for name in ("Node", "VariableNode", "Edge"):
            with self.subTest(name=name):
                self.assertNotIn(name, vars(computation_module))

    def test_structural_state_is_gone(self):
        value = ts.Variable([2.0], requires_grad=True)
        computation = Computation(ts.sum(value * value))

        for name in (
            "_all_nodes",
            "_node_masks",
            "_boundary_nodes",
            "_nodes",
            "_output_bit",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(computation, name))

    def test_no_edge_inspection_or_node_classification(self):
        source = inspect.getsource(computation_module)
        self.assertNotIn("_in_edges", source)
        self.assertNotIn("_out_edges", source)
        # Deriving an execution view by classifying graph vertices is the
        # compiler's job now.
        self.assertNotIn("VariableNode", source)

    def test_execution_state_is_the_compiled_domain(self):
        value = ts.Variable([2.0], requires_grad=True)
        computation = Computation(ts.sum(value * value))

        for name in (
            "output",
            "_output_slot",
            "_variables",
            "_variable_slots",
            "_leaf_slots",
            "_instructions",
            "_view_slots",
            "_view_instructions",
            "_fusions",
            "_fusion_starts",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(computation, name))


class CompiledViewTests(unittest.TestCase):
    """Execution views are resolved during compilation."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_compiler_resolves_the_views(self):
        shared = ts.Variable([2.0])
        only_first = ts.Variable([3.0])
        first = shared * only_first
        second = shared + 1.0

        compiler = Compiler((first, second))
        compiler.compile()

        self.assertEqual(len(compiler.view_slots), 2)
        self.assertEqual(len(compiler.view_instructions), 2)
        self.assertEqual(len(compiler.view_nodes), 2)
        self.assertEqual(
            [i.operation.name for i in compiler.view_instructions[0]], ["mul"]
        )
        self.assertEqual(
            [i.operation.name for i in compiler.view_instructions[1]], ["add"]
        )
        self.assertIn(compiler.variable_slots[only_first], compiler.view_slots[0])
        self.assertNotIn(
            compiler.variable_slots[only_first], compiler.view_slots[1]
        )

    def test_computation_takes_the_view_it_was_given(self):
        shared = ts.Variable([2.0])
        first = shared * 3.0
        second = shared + 1.0

        compiler = Compiler((first, second))
        compiler.compile()
        one, two = Computation._from_compiler(compiler)

        # The views are the compiler's objects, not recomputed here.
        self.assertIs(one._view_slots, compiler.view_slots[0])
        self.assertIs(two._view_slots, compiler.view_slots[1])
        self.assertIs(one._view_instructions, compiler.view_instructions[0])
        self.assertIs(two._view_instructions, compiler.view_instructions[1])
        self.assertIs(one._view_nodes, compiler.view_nodes[0])
        self.assertIs(two._view_nodes, compiler.view_nodes[1])

    def test_select_view_is_gone(self):
        self.assertFalse(hasattr(Computation, "_select_view"))

    def test_multi_output_still_shares_one_program(self):
        shared = ts.Variable([2.0], requires_grad=True)
        one, two = Computation.from_outputs([shared * 3.0, shared + 1.0])

        for name in (
            "_variables",
            "_variable_slots",
            "_leaf_slots",
            "_instructions",
            "_fusions",
            "_fusion_starts",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(one, name), getattr(two, name))
        self.assertIsNot(one._view_instructions, two._view_instructions)
        self.assertIsNot(one.output, two.output)
        self.assertNotEqual(one._output_slot, two._output_slot)
        self.assertEqual(one.forward().tolist(), [6.0])
        self.assertEqual(two.forward().tolist(), [3.0])

    def test_nodes_property_stays_behaviourally_compatible(self):
        value = ts.Variable([2.0])
        output = (value + 1.0) * 3.0
        compiler = Compiler((output,))
        compiler.compile()

        computation = Computation(output)
        nodes = computation.nodes

        self.assertEqual(nodes, list(compiler.view_nodes[0]))
        self.assertIs(nodes[-1], output.node)
        # Still an independent list per access.
        nodes.clear()
        self.assertEqual(computation.nodes, list(compiler.view_nodes[0]))

        computation.release()
        with self.assertRaisesRegex(RuntimeError, "released"):
            _ = computation.nodes


class GraphStructuralMetadataTests(unittest.TestCase):
    """The graph layer takes its structure from the compiler."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _model():
        class Model(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([3.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        return Model()

    def test_graph_does_not_read_structural_state_from_computation(self):
        source = inspect.getsource(graph_module)
        for name in ("_all_nodes", "_boundary_nodes", "_node_masks", "_in_edges"):
            with self.subTest(name=name):
                self.assertNotIn(name, source)
        self.assertFalse(hasattr(ts.Graph, "_capture"))

    def test_graph_metadata_matches_the_compilation(self):
        model = self._model()
        value = ts.Variable([2.0])
        model(value)

        compiler = Compiler(
            (model._materialize_state().computations[0].output,),
            boundaries=(value,),
        )
        compiler.compile()

        self.assertEqual(model.nodes, list(compiler.nodes))
        self.assertEqual(model.edges, list(compiler.edges))

    def test_graph_compiles_once_per_trace(self):
        model = self._model()
        value = ts.Variable([2.0])
        original = Compiler.compile
        compilations = []

        def counted(self):
            compilations.append(self)
            return original(self)

        with patch.object(Compiler, "compile", counted):
            model(value)
            nodes = model.nodes
            edges = model.edges

        self.assertEqual(len(compilations), 1)
        self.assertEqual(len(nodes), 7)
        self.assertTrue(edges)

    def test_graph_boundaries_still_stop_at_inputs(self):
        model = self._model()
        source = ts.Variable([2.0])
        value = source * 5.0
        model(value)

        # The traced program starts at the Graph's input, not behind it.
        self.assertNotIn(source.node, model.nodes)
        self.assertIn(value.node, model.nodes)
        computation = model._materialize_state().computations[0]
        self.assertEqual(
            [i.operation.name for i in computation._instructions],
            ["mul", "add"],
        )

    def test_graph_replay_and_gradients_are_unchanged(self):
        model = self._model()
        value = ts.Variable([2.0])
        loss = ts.sum(model(value))

        ts.backward(loss)

        self.assertEqual(model.weight.grad.tolist(), [2.0])
        self.assertEqual(model.bias.grad.tolist(), [1.0])
        self.assertEqual(
            model._materialize_state().computations[0].forward().tolist(), [7.0]
        )


if __name__ == "__main__":
    unittest.main()
