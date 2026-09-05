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
from tensors.graph.computation.compiler import Compiler
from tensors.graph.computation.instruction import Instruction
from tensors.graph.node import VariableNode
from tensors.graph.state import reset_graph_state


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

        compiler = Compiler((output,))
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

        compiler = Compiler((output,))
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

        compiler = Compiler((output,))
        compiler.compile()

        self.assertEqual(
            compiler.variables,
            tuple(
                node.variable
                for node in compiler.nodes
                if isinstance(node, VariableNode)
            ),
        )
        self.assertEqual(
            compiler.variable_slots,
            {variable: index for index, variable in enumerate(compiler.variables)},
        )
        self.assertEqual(
            sorted(compiler.leaf_slots),
            sorted(
                compiler.variable_slots[variable] for variable in (left, right)
            ),
        )
        self.assertEqual(
            compiler.output_slots, (compiler.variable_slots[output],)
        )

    def test_compiler_emits_instructions_over_slots(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = left * right

        compiler = Compiler((output,))
        instruction, = compiler.compile()

        slots = compiler.variable_slots
        self.assertEqual(
            instruction.input_slots, (slots[left], slots[right])
        )
        self.assertEqual(instruction.output_slot, slots[output])

    def test_compiler_respects_boundaries(self):
        value = ts.Variable([2.0])
        hidden = value * 3.0
        output = hidden + 1.0

        compiler = Compiler((output,), boundaries=(hidden,))
        instructions = compiler.compile()

        self.assertEqual(
            [instruction.operation.name for instruction in instructions], ["add"]
        )
        self.assertIn(hidden.node, compiler.boundary_nodes)
        self.assertIn(compiler.variable_slots[hidden], compiler.leaf_slots)
        self.assertNotIn(value.node, compiler.nodes)

    def test_compiler_records_per_output_reachability(self):
        shared = ts.Variable([2.0])
        only_first = ts.Variable([3.0])
        first = shared * only_first
        second = shared + 1.0

        compiler = Compiler((first, second))
        compiler.compile()

        masks = dict(zip(compiler.nodes, compiler.node_masks))
        self.assertEqual(masks[first.node], 0b01)
        self.assertEqual(masks[second.node], 0b10)
        self.assertEqual(masks[shared.node], 0b11)
        self.assertEqual(masks[only_first.node], 0b01)
        self.assertEqual(
            compiler.output_slots,
            (compiler.variable_slots[first], compiler.variable_slots[second]),
        )

    def test_compiler_validates_outputs(self):
        with self.assertRaisesRegex(ValueError, "at least one output"):
            Compiler(())
        with self.assertRaisesRegex(TypeError, "graph node"):
            Compiler((ts.Tensor([1.0]),))

    def test_compiler_plans_no_fusion(self):
        value = ts.Variable(ts.full((4_096,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)

        compiler = Compiler((output,))
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
        for name in ("Compiler", "Instruction", "validate_outputs"):
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
        compiler = Compiler((output,))
        compiler.compile()

        computation = Computation(output)

        self.assertEqual(computation._variables, compiler.variables)
        self.assertEqual(computation._variable_slots, compiler.variable_slots)
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
        self.assertIs(one._variables, two._variables)
        self.assertIs(one._variable_slots, two._variable_slots)
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
        self.assertIn(one._variable_slots[only_first], one._view_slots)
        self.assertNotIn(two._variable_slots[only_first], two._view_slots)
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
