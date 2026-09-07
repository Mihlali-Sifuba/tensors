import ast
import inspect
import pathlib
import unittest
from importlib import import_module
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation
from tensors.graph import graph as graph_module
from tensors.graph.computation import computation as computation_module
from tensors.graph.computation.compiler import Compiler
from tensors.graph.expression import UnsupportedStructuralExpression
from tensors.graph.node import VariableNode
from tensors.graph.state import reset_graph_state
from tensors.math.concat import Concat
from tensors.math.convolution import ConvND
from tensors.math.cross_entropy import CrossEntropy
from tensors.math.stack import Stack

# The math package binds each public function over its module, so the module
# holding the coercion boundary has to be named directly.
concat_module = import_module("tensors.math.concat")
stack_module = import_module("tensors.math.stack")


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


def forward_sources() -> dict[str, str]:
    """Return the source of every forward() written below the graph layer."""
    package = pathlib.Path(ts.__file__).parent
    sources: dict[str, str] = {}
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if "graph" in relative.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "forward"
                ):
                    owner = f"{relative.as_posix()}:{node.name}"
                    sources[owner] = ast.get_source_segment(text, item) or ""
    return sources


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
        # A Computation identifies each slot by the vertex naming its value
        # and materializes that value, but it never reads topology: deriving
        # an execution view from the graph is the compiler's job.
        for name in (
            "_in_edges",
            "_out_edges",
            "producer",
            "operand_nodes",
            "result_node",
            "Edge",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, source)

    def test_execution_state_is_the_compiled_domain(self):
        value = ts.Variable([2.0], requires_grad=True)
        computation = Computation(ts.sum(value * value))

        for name in (
            "output",
            "_output_slot",
            "_variable_nodes",
            "_node_slots",
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

        compiler = Compiler((first.node, second.node))
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
        slot = compiler.node_slots[only_first.node]
        self.assertIn(slot, compiler.view_slots[0])
        self.assertNotIn(slot, compiler.view_slots[1])

    def test_computation_takes_the_view_it_was_given(self):
        shared = ts.Variable([2.0])
        first = shared * 3.0
        second = shared + 1.0

        compiler = Compiler((first.node, second.node))
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
            "_variable_nodes",
            "_node_slots",
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
        compiler = Compiler((output.node,))
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
            (model._materialize_state().computations[0].output.node,),
            boundaries=(value.node,),
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
            eager = len(compilations)
            nodes = model.nodes
            traced = len(compilations)
            edges = model.edges

        # Each eager operation compiles its own one-instruction fragment as
        # it runs.
        self.assertEqual(
            [len(compiler.instructions) for compiler in compilations[:eager]],
            [1, 1],
        )
        # The trace compiles one plan spanning them, and reading further
        # structural metadata from it compiles nothing more.
        self.assertEqual(traced, eager + 1)
        self.assertEqual(len(compilations[-1].instructions), 2)
        self.assertEqual(len(compilations), traced)
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


class TensorLayerIndependenceTests(unittest.TestCase):
    """The numerical layer stays unaware of the graph layer above it."""

    def test_tensor_names_nothing_from_the_graph_layer(self):
        import tensors.tensor as tensor_module

        source = inspect.getsource(tensor_module)
        for name in (
            "graph",
            "VariableNode",
            "UnsupportedStructuralExpression",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, source)

    def test_tensor_imports_nothing_from_the_graph_layer(self):
        import tensors.tensor as tensor_module

        for name in runtime_imported_names(tensor_module):
            with self.subTest(name=name):
                self.assertNotIn("graph", name)

    def test_tensor_treats_a_vertex_as_ordinary_unsupported_data(self):
        from tensors.graph.expression import UnsupportedStructuralExpression
        from tensors.graph.node import VariableNode

        with self.assertRaisesRegex(TypeError, "Unsupported data type"):
            ts.Tensor(VariableNode())
        with self.assertRaises(TypeError) as caught:
            ts.Tensor(VariableNode())
        # Constructing a Tensor from an unsupported object is a value error
        # like any other; only the graph layer knows what a vertex means.
        self.assertNotIsInstance(
            caught.exception, UnsupportedStructuralExpression
        )


class OperationForwardIndependenceTests(unittest.TestCase):
    """A numerical forward() runs on values it is handed, not on operands."""

    #: What a forward() would have to name to know about the graph above it.
    GRAPH_NAMES = (
        "graph",
        "as_tensor_operand",
        "as_graph_operand",
        "is_graph_operand",
        "apply_operation",
        "record_structurally",
        "structural_node",
        "UnsupportedStructuralExpression",
        "VariableNode",
    )

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_no_forward_names_the_graph_layer(self):
        sources = forward_sources()

        # The two that reached upward for coercion, and enough of the rest
        # to show the scan really covers the numerical layer.
        for owner in (
            "math/concat.py:Concat",
            "math/stack.py:Stack",
            "math/convolution.py:ConvND",
            "math/arg_extrema.py:_ArgExtremum",
            "math/where.py:Where",
            "ops/add.py:Add",
        ):
            with self.subTest(forward=owner):
                self.assertIn(owner, sources)

        for owner, source in sources.items():
            for name in self.GRAPH_NAMES:
                with self.subTest(forward=owner, name=name):
                    self.assertNotIn(name, source)

    def test_concat_coerces_at_the_public_function(self):
        coercions = []
        original = concat_module.as_tensor_operand

        def counted(value, **options):
            coercions.append(value)
            return original(value, **options)

        with patch.object(concat_module, "as_tensor_operand", counted):
            executed = Concat().forward(ts.Tensor([1.0]), ts.Tensor([2.0]))
            self.assertEqual(coercions, [])
            through_function = ts.concat([[1.0], ts.Tensor([2.0])])

        self.assertEqual(len(coercions), 2)
        self.assertEqual(executed.tolist(), [1.0, 2.0])
        self.assertEqual(through_function.tolist(), [1.0, 2.0])

    def test_stack_coerces_at_the_public_function(self):
        coercions = []
        original = stack_module.as_tensor_operand

        def counted(value, **options):
            coercions.append(value)
            return original(value, **options)

        with patch.object(stack_module, "as_tensor_operand", counted):
            executed = Stack().forward(ts.Tensor([1.0]), ts.Tensor([2.0]))
            self.assertEqual(coercions, [])
            through_function = ts.stack([[1.0], ts.Tensor([2.0])])

        self.assertEqual(len(coercions), 2)
        self.assertEqual(executed.tolist(), [1.0, 2.0])
        self.assertEqual(through_function.shape, (2, 1))

    def test_forward_is_reached_with_tensors_only(self):
        operands = []

        def watching(original):
            def forward(self, *tensors):
                operands.extend(tensors)
                return original(self, *tensors)

            return forward

        with patch.object(Concat, "forward", watching(Concat.forward)):
            ts.concat([[1.0], ts.Tensor([2.0]), ts.Variable([3.0]).data])
        with patch.object(Stack, "forward", watching(Stack.forward)):
            ts.stack([[1.0], ts.Tensor([2.0])])

        self.assertEqual(len(operands), 5)
        for operand in operands:
            with self.subTest(operand=operand):
                self.assertIsInstance(operand, ts.Tensor)

    def test_a_vertex_is_rejected_before_forward_runs(self):
        executions = []

        def counted(self, *tensors):
            executions.append(tensors)
            raise AssertionError("a vertex must never reach forward()")

        calls = {
            "conv1d": (
                ConvND,
                lambda: ts.conv1d(VariableNode(), ts.ones((1, 1, 2))),
            ),
            "cross_entropy": (
                CrossEntropy,
                lambda: ts.cross_entropy(
                    VariableNode(), ts.Tensor([0], dtype=ts.int64)
                ),
            ),
        }
        for name, (operation, call) in calls.items():
            with self.subTest(operation=name):
                with patch.object(operation, "forward", counted):
                    with self.assertRaises(UnsupportedStructuralExpression):
                        call()
        self.assertEqual(executions, [])

    def test_variable_execution_and_autograd_are_unchanged(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])

        joined = ts.concat([left, right])
        stacked = ts.stack([left, right], axis=1)

        self.assertIsInstance(joined, ts.Variable)
        self.assertIsInstance(stacked, ts.Variable)
        self.assertEqual(joined.data.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stacked.shape, (2, 2))

        ts.backward(ts.sum(ts.concat([left, right]) ** 2.0))

        self.assertEqual(left.grad.tolist(), [2.0, 4.0])
        self.assertEqual(right.grad.tolist(), [6.0, 8.0])

    def test_a_supported_structural_operation_still_records(self):
        inputs = VariableNode()
        weight = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")

        output = ts.relu(inputs @ weight)

        self.assertIsInstance(output, VariableNode)
        self.assertFalse(output.is_bound)
        self.assertEqual(output.producer.operation.name, "relu")


if __name__ == "__main__":
    unittest.main()
