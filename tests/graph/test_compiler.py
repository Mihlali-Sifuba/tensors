import ast
import inspect
import subprocess
import sys
import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph import computation as computation_package
from tensors.graph.computation import compiler as compiler_module
from tensors.graph.computation import computation as computation_module
from tensors.graph.computation import instruction as instruction_module
from tensors.graph.computation.compiler import (
    Compiler, resolve_boundaries, resolve_outputs,
)
from tensors.graph.computation.instruction import Instruction
from tensors.graph.edge import Edge
from tensors.graph.node import OperationNode, VariableNode
from tensors.graph.state import reset_graph_state
from tensors.ops import Add, Mul


class InstructionModuleTests(unittest.TestCase):
    """One executable operation invocation has its own module."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_instruction_is_defined_by_the_instruction_module(self):
        self.assertEqual(
            Instruction.__module__, "tensors.graph.computation.instruction"
        )
        self.assertIs(instruction_module.Instruction, Instruction)

    def test_computation_module_no_longer_defines_instruction(self):
        self.assertNotIn("Instruction", vars(computation_module))

    def test_instruction_semantics_are_unchanged(self):
        computation = Computation(ts.Variable([2.0]) * ts.Variable([3.0]))
        instruction = computation._instructions[0]

        self.assertIsInstance(instruction, Instruction)
        self.assertEqual(
            Instruction.__slots__, ("operation", "input_slots", "output_slot")
        )
        with self.assertRaises(AttributeError):
            instruction.output_slot = 0


class CompilerResponsibilityTests(unittest.TestCase):
    """The compiler translates graph structure into an instruction program."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_compile_returns_the_instruction_sequence(self):
        value = ts.Variable([2.0])
        output = ts.sum(value * 3.0 + 1.0)

        compiler = Compiler((output.node,))
        instructions = compiler.compile()

        self.assertIs(instructions, compiler.instructions)
        self.assertEqual(
            [instruction.operation.name for instruction in instructions],
            ["mul", "add", "sum"],
        )
        for instruction in instructions:
            self.assertIsInstance(instruction, Instruction)

    def test_compiler_owns_dependency_analysis(self):
        value = ts.Variable([2.0])
        output = (value + 1.0) * 3.0

        compiler = Compiler((output.node,))
        compiler.compile()

        # The traversal is dependency-first and reaches the output last.
        self.assertIs(compiler.nodes[-1], output.node)
        self.assertIn(value.node, compiler.nodes)
        self.assertEqual(len(compiler.nodes), len(compiler.node_masks))
        # A single output reaches everything it was traced from.
        self.assertEqual(set(compiler.node_masks), {1})

    def test_compiler_owns_slot_assignment(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = left * right

        compiler = Compiler((output.node,))
        compiler.compile()

        self.assertEqual(
            compiler.variable_nodes,
            tuple(
                node
                for node in compiler.nodes
                if isinstance(node, VariableNode)
            ),
        )
        self.assertEqual(
            compiler.node_slots,
            {node: index for index, node in enumerate(compiler.variable_nodes)},
        )
        self.assertEqual(
            sorted(compiler.leaf_slots),
            sorted(
                compiler.node_slots[value.node] for value in (left, right)
            ),
        )
        self.assertEqual(
            compiler.output_slots, (compiler.node_slots[output.node],)
        )

    def test_compiler_emits_instructions_over_slots(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = left * right

        compiler = Compiler((output.node,))
        instruction, = compiler.compile()

        slots = compiler.node_slots
        self.assertEqual(
            instruction.input_slots, (slots[left.node], slots[right.node])
        )
        self.assertEqual(instruction.output_slot, slots[output.node])

    def test_compiler_respects_boundaries(self):
        value = ts.Variable([2.0])
        hidden = value * 3.0
        output = hidden + 1.0

        compiler = Compiler((output.node,), boundaries=(hidden.node,))
        instructions = compiler.compile()

        self.assertEqual(
            [instruction.operation.name for instruction in instructions], ["add"]
        )
        self.assertIn(hidden.node, compiler.boundary_nodes)
        self.assertIn(compiler.node_slots[hidden.node], compiler.leaf_slots)
        self.assertNotIn(value.node, compiler.nodes)

    def test_compiler_records_per_output_reachability(self):
        shared = ts.Variable([2.0])
        only_first = ts.Variable([3.0])
        first = shared * only_first
        second = shared + 1.0

        compiler = Compiler((first.node, second.node))
        compiler.compile()

        masks = dict(zip(compiler.nodes, compiler.node_masks))
        self.assertEqual(masks[first.node], 0b01)
        self.assertEqual(masks[second.node], 0b10)
        self.assertEqual(masks[shared.node], 0b11)
        self.assertEqual(masks[only_first.node], 0b01)
        self.assertEqual(
            compiler.output_slots,
            (
                compiler.node_slots[first.node],
                compiler.node_slots[second.node],
            ),
        )

    def test_compiler_validates_outputs(self):
        with self.assertRaisesRegex(ValueError, "at least one output"):
            Compiler(())
        with self.assertRaisesRegex(TypeError, "must be a VariableNode"):
            Compiler((ts.Tensor([1.0]),))

    def test_runtime_outputs_are_resolved_to_their_vertices(self):
        value = ts.Variable([2.0])
        output = value * 3.0

        self.assertEqual(resolve_outputs((output,)), (output.node,))
        self.assertEqual(resolve_boundaries((value,)), (value.node,))
        self.assertEqual(resolve_boundaries(()), ())
        with self.assertRaisesRegex(ValueError, "at least one output"):
            resolve_outputs(())
        with self.assertRaisesRegex(TypeError, "output must have a graph node"):
            resolve_outputs((ts.Tensor([1.0]),))
        with self.assertRaisesRegex(
            TypeError, "boundary must have a graph node"
        ):
            resolve_boundaries((ts.Tensor([1.0]),))

    def test_compiler_plans_no_fusion(self):
        value = ts.Variable(ts.full((4_096,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)

        compiler = Compiler((output.node,))
        compiler.compile()

        # Fusion is an optimization over a compiled program, not part of
        # compiling one.
        for name in ("fusions", "fusion_starts", "plan_fusions"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(compiler, name))
        self.assertNotIn("fusion", self._imports_of(compiler_module))

    def test_compiler_does_not_depend_on_computation(self):
        self.assertNotIn("Computation", vars(compiler_module))
        imported = self._imports_of(compiler_module)
        self.assertNotIn("computation", imported)
        self.assertNotIn("Computation", imported)
        self.assertNotIn("tensors.graph.computation.computation", imported)

    def test_computation_no_longer_compiles_itself(self):
        for name in (
            "_validate_outputs",
            "_dependency_plan",
            "_dependency_order",
            "_compile_execution_plan",
            "_initialize_plan",
            "_initialize_shared_view",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(Computation, name))

        source = inspect.getsource(computation_module)
        # Graph traversal reads edges and instruction emission constructs
        # Instructions. Neither happens here any more.
        self.assertNotIn("_in_edges", source)
        self.assertNotIn("Instruction(", source)
        self.assertNotIn("edge", source)

    def test_modules_import_first_without_a_cycle(self):
        for name in ("instruction", "compiler", "computation"):
            with self.subTest(module=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import tensors.graph.computation.{name}",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_compiler_and_instruction_are_not_public_api(self):
        for name in ("Compiler", "Instruction", "resolve_outputs"):
            with self.subTest(name=name):
                self.assertNotIn(name, computation_package.__all__)
                self.assertNotIn(name, ts.graph.__all__)
                self.assertFalse(hasattr(computation_package, name))
                self.assertFalse(hasattr(ts.graph, name))
                self.assertFalse(hasattr(ts, name))

    @staticmethod
    def _imports_of(module):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        return imported


class NodeIdentityCompilationTests(unittest.TestCase):
    """Slots are numbered by vertex, so a value need not exist to compile."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _pending_graph():
        """Return the vertices of ``c = a + b`` and ``d = c * a``, c and d
        unbound."""
        a = ts.Variable([1.0, 2.0, 3.0], name="a")
        b = ts.Variable([4.0, 5.0, 6.0], name="b")
        add = OperationNode(Add())
        multiply = OperationNode(Mul())
        c = VariableNode()
        d = VariableNode()
        Edge(a.node, add, label="input_0")
        Edge(b.node, add, label="input_1")
        Edge(add, c, label="result")
        Edge(c, multiply, label="input_0")
        Edge(a.node, multiply, label="input_1")
        Edge(multiply, d, label="result")
        return a, b, c, d

    def test_an_unbound_result_vertex_receives_a_slot(self):
        a, b, c, _ = self._pending_graph()

        compiler = Compiler((c,))
        compiler.compile()

        self.assertFalse(c.is_bound)
        self.assertEqual(compiler.variable_nodes, (a.node, b.node, c))
        self.assertEqual(compiler.node_slots, {a.node: 0, b.node: 1, c: 2})
        self.assertEqual(compiler.output_slots, (2,))

    def test_bound_leaves_and_an_unbound_result_compile_together(self):
        a, b, c, d = self._pending_graph()

        compiler = Compiler((d,))
        instructions = compiler.compile()

        self.assertEqual(
            [instruction.operation.name for instruction in instructions],
            ["add", "mul"],
        )
        self.assertTrue(a.node.is_bound and b.node.is_bound)
        self.assertFalse(c.is_bound or d.is_bound)

    def test_instruction_slots_are_derived_from_vertex_identity(self):
        a, b, c, d = self._pending_graph()

        compiler = Compiler((d,))
        addition, product = compiler.compile()
        slots = compiler.node_slots

        self.assertEqual(
            addition.input_slots, (slots[a.node], slots[b.node])
        )
        self.assertEqual(addition.output_slot, slots[c])
        self.assertEqual(product.input_slots, (slots[c], slots[a.node]))
        self.assertEqual(product.output_slot, slots[d])
        for node in slots:
            with self.subTest(node=node):
                self.assertIsInstance(node, VariableNode)

    def test_leaf_slots_are_the_vertices_nothing_produces(self):
        a, b, c, d = self._pending_graph()

        compiler = Compiler((d,))
        compiler.compile()
        slots = compiler.node_slots

        self.assertEqual(
            sorted(compiler.leaf_slots), sorted((slots[a.node], slots[b.node]))
        )
        self.assertNotIn(slots[c], compiler.leaf_slots)
        self.assertNotIn(slots[d], compiler.leaf_slots)


    def test_multiple_unbound_outputs_each_resolve_their_own_view(self):
        a, b, c, d = self._pending_graph()

        compiler = Compiler((c, d))
        compiler.compile()
        slots = compiler.node_slots

        self.assertEqual(compiler.output_slots, (slots[c], slots[d]))
        self.assertEqual(
            [i.operation.name for i in compiler.view_instructions[0]], ["add"]
        )
        self.assertEqual(
            [i.operation.name for i in compiler.view_instructions[1]],
            ["add", "mul"],
        )
        self.assertNotIn(slots[d], compiler.view_slots[0])
        self.assertIn(slots[d], compiler.view_slots[1])
        self.assertNotIn(d, compiler.view_nodes[0])

    def test_a_boundary_vertex_still_stops_the_traversal(self):
        a, b, c, d = self._pending_graph()

        compiler = Compiler((d,), boundaries=(c,))
        instructions = compiler.compile()
        slots = compiler.node_slots

        self.assertEqual(
            [instruction.operation.name for instruction in instructions],
            ["mul"],
        )
        self.assertIn(c, compiler.boundary_nodes)
        self.assertIn(slots[c], compiler.leaf_slots)
        self.assertNotIn(b.node, compiler.nodes)
        self.assertIn(a.node, compiler.nodes)

    def test_a_materialized_graph_still_compiles_to_the_same_slots(self):
        value = ts.Variable([2.0])
        output = ts.sum(value * 3.0)

        compiler = Compiler((output.node,))
        compiler.compile()

        self.assertEqual(
            compiler.node_slots,
            {
                node: slot
                for slot, node in enumerate(compiler.variable_nodes)
            },
        )
        # Binding a value changes nothing about the program: a compilation
        # keeps no projection of its slots onto runtime Variables.
        for name in ("variables", "variable_slots"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(compiler, name))

    def test_the_compiler_never_reads_a_runtime_value(self):
        value_reads = {"variable", "operands", "result", "data", "grad"}

        tree = ast.parse(inspect.getsource(compiler_module))
        for definition in ast.walk(tree):
            if (
                not isinstance(definition, ast.ClassDef)
                or definition.name != "Compiler"
            ):
                continue
            self.assertEqual(
                [
                    node.attr
                    for node in ast.walk(definition)
                    if isinstance(node, ast.Attribute)
                    and node.attr in value_reads
                ],
                [],
            )


class CompiledComputationTests(unittest.TestCase):
    """A Computation consumes a compiled program and executes it."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_single_output_construction_is_unchanged(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value + 1.0)

        computation = Computation(output)

        self.assertIs(computation.output, output)
        self.assertEqual(computation.forward().tolist(), [5.0])
        computation.backward()
        self.assertEqual(value.grad.tolist(), [4.0])
        self.assertEqual(
            [instruction.operation.name
             for instruction in computation._instructions],
            ["mul", "add", "sum"],
        )

    def test_computation_stores_what_the_compiler_produced(self):
        value = ts.Variable([2.0])
        output = ts.sum(value * 3.0)
        compiler = Compiler((output.node,))
        compiler.compile()

        computation = Computation(output)

        self.assertEqual(computation._variable_nodes, compiler.variable_nodes)
        self.assertEqual(computation._node_slots, compiler.node_slots)
        self.assertEqual(computation._leaf_slots, compiler.leaf_slots)
        self.assertEqual(computation._output_slot, compiler.output_slots[0])
        self.assertEqual(
            [
                (i.operation.name, i.input_slots, i.output_slot)
                for i in computation._instructions
            ],
            [
                (i.operation.name, i.input_slots, i.output_slot)
                for i in compiler.instructions
            ],
        )

    def test_multi_output_shares_one_compilation(self):
        shared = ts.Variable([2.0], requires_grad=True)
        first = shared * 3.0
        second = shared + 1.0

        one, two = Computation.from_outputs([first, second])

        # One compilation: the program and its slot map are the same objects.
        self.assertIs(one._instructions, two._instructions)
        self.assertIs(one._variable_nodes, two._variable_nodes)
        self.assertIs(one._node_slots, two._node_slots)
        self.assertIs(one._fusions, two._fusions)
        self.assertIs(one._fusion_starts, two._fusion_starts)
        # Each view still executes only its own output.
        self.assertEqual(one.forward().tolist(), [6.0])
        self.assertEqual(two.forward().tolist(), [3.0])
        self.assertIs(one.output, first)
        self.assertIs(two.output, second)

    def test_per_output_views_select_their_own_slice(self):
        shared = ts.Variable([2.0])
        only_first = ts.Variable([3.0])
        first = shared * only_first
        second = shared + 1.0

        one, two = Computation.from_outputs([first, second])

        self.assertEqual(
            [i.operation.name for i in one._view_instructions], ["mul"]
        )
        self.assertEqual(
            [i.operation.name for i in two._view_instructions], ["add"]
        )
        self.assertIn(one._node_slots[only_first.node], one._view_slots)
        self.assertNotIn(two._node_slots[only_first.node], two._view_slots)
        self.assertIn(first.node, one.nodes)
        self.assertNotIn(first.node, two.nodes)

    def test_boundaries_still_stop_compilation(self):
        value = ts.Variable([2.0], requires_grad=True)
        hidden = value * 3.0
        output = hidden + 1.0

        computation, = Computation.from_outputs([output], boundaries=[hidden])

        self.assertEqual(
            [i.operation.name for i in computation._instructions], ["add"]
        )
        self.assertEqual(computation.forward().tolist(), [7.0])

    def test_forward_and_backward_survive_a_release_and_rebuild(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        computation = Computation(output)
        computation.release()
        rebuilt = Computation(output)

        self.assertEqual(rebuilt.forward().tolist(), [4.0])
        rebuilt.backward()
        self.assertEqual(value.grad.tolist(), [4.0])

    def test_fusion_metadata_still_accompanies_the_program(self):
        value = ts.Variable(ts.full((4_096,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)

        computation = Computation(output)

        self.assertEqual(list(computation._fusions), [0])
        end, steps, _ = computation._fusions[0]
        self.assertEqual(end, 2)
        self.assertEqual(
            [step[0] for step in steps], ["multiply", "sin", "add"]
        )
        self.assertEqual(computation._fusion_starts, {2: 0})

    def test_execution_agrees_across_backends(self):
        expected = None
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                reset_graph_state()
                value = ts.Variable(ts.full((512,), 0.5), requires_grad=True)
                output = ts.sum(ts.tanh(ts.sin(value * 1.5) + 0.25))
                computation = Computation(output)
                replayed = computation.forward().tolist()
                computation.backward()
                # Reduction order differs between backends, so compare to
                # within floating-point tolerance rather than bit for bit.
                result = (
                    [round(item, 9) for item in replayed],
                    [round(item, 9) for item in value.grad.tolist()],
                )
                if expected is None:
                    expected = result
                self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
