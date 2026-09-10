import unittest

import tensors as ts
from tensors.graph import (
    Computation, Edge, OperationNode, UnboundVariableNodeError, VariableNode,
)
from tensors.graph.computation.compiler import Compiler
from tensors.graph.state import get_graph_state, reset_graph_state
from tensors.ops import Add


class VariableNodeLifecycleTests(unittest.TestCase):
    """A vertex is the graph identity of a value, not a wrapper around one."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_vertex_can_be_created_before_its_value_exists(self):
        node = VariableNode()

        self.assertFalse(node.is_bound)
        self.assertEqual(node.label, "var")
        self.assertEqual(node.inputs, [])
        self.assertIsNone(node.producer)
        with self.assertRaisesRegex(
            UnboundVariableNodeError, "has not been materialized"
        ):
            node.variable

    def test_a_leaf_binds_its_vertex_when_the_variable_is_constructed(self):
        variable = ts.Variable([1.0], name="weight")

        self.assertIsInstance(variable.node, VariableNode)
        self.assertTrue(variable.node.is_bound)
        self.assertIs(variable.node.variable, variable)

    def test_a_variable_can_be_materialized_against_an_existing_vertex(self):
        node = VariableNode()

        variable = ts.Variable([2.0], name="late", node=node)

        self.assertTrue(node.is_bound)
        self.assertIs(node.variable, variable)
        self.assertIs(variable.node, node)
        self.assertEqual(variable.name, "late")

    def test_materialize_creates_the_variable_the_vertex_names(self):
        node = VariableNode()

        variable = node.materialize([3.0, 4.0], "result", requires_grad=False)

        self.assertIs(node.variable, variable)
        self.assertIs(variable.node, node)
        self.assertEqual(variable.name, "result")
        self.assertFalse(variable.requires_grad)
        self.assertEqual(variable.data.tolist(), [3.0, 4.0])

    def test_the_registry_records_a_vertex_before_its_value_exists(self):
        state = get_graph_state()

        node = state.add_variable_node()

        self.assertEqual(state.nodes, [node])
        self.assertFalse(node.is_bound)

    def test_a_vertex_cannot_be_bound_twice(self):
        node = VariableNode()
        first = node.materialize([1.0], "first")

        with self.assertRaisesRegex(RuntimeError, "already bound"):
            node.materialize([2.0], "second")
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            node.bind(first)
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            ts.Variable([3.0], node=node)
        self.assertIs(node.variable, first)

    def test_a_variable_cannot_be_bound_to_a_second_vertex(self):
        variable = ts.Variable([1.0], name="owned")
        original = variable.node

        with self.assertRaisesRegex(RuntimeError, "already bound"):
            VariableNode().bind(variable)
        self.assertIs(variable.node, original)
        self.assertIs(original.variable, variable)

    def test_a_variable_rejects_a_node_that_is_not_a_variable_vertex(self):
        with self.assertRaisesRegex(TypeError, "node must be a VariableNode"):
            ts.Variable([1.0], node=OperationNode(Add()))


class UnmaterializedGraphStructureTests(unittest.TestCase):
    """Structure is readable before the values it names are calculated."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _record_addition():
        """Return the operands and the unbound result vertex of ``a + b``."""
        a = ts.Variable([1.0, 2.0, 3.0], name="a")
        b = ts.Variable([4.0, 5.0, 6.0], name="b")
        operation = OperationNode(Add())
        result = VariableNode()
        Edge(a.node, operation, label="input_0")
        Edge(b.node, operation, label="input_1")
        Edge(operation, result, label="result")
        return a, b, operation, result

    def test_connectivity_is_complete_before_the_result_is_materialized(self):
        a, b, operation, result = self._record_addition()

        self.assertIs(result.producer, operation)
        self.assertEqual(operation.operand_nodes, (a.node, b.node))
        self.assertEqual(operation.operands, (a, b))
        self.assertIs(operation.result_node, result)
        self.assertEqual(result.inputs, [operation])
        self.assertEqual(a.node.outputs, [operation])

    def test_reading_an_unmaterialized_result_is_a_lifecycle_error(self):
        _, _, operation, result = self._record_addition()

        self.assertFalse(result.is_bound)
        with self.assertRaises(UnboundVariableNodeError):
            operation.result

    def test_the_variable_is_materialized_after_the_operation_runs(self):
        a, b, operation, result = self._record_addition()

        value = operation.operation.forward(a.data, b.data)
        c = result.materialize(value, "c")

        self.assertTrue(result.is_bound)
        self.assertIs(operation.result, c)
        self.assertIs(c.node, result)
        self.assertEqual(c.data.tolist(), [5.0, 7.0, 9.0])

    def test_a_graph_named_before_execution_still_compiles_and_replays(self):
        a, b, operation, result = self._record_addition()
        value = operation.operation.forward(a.data, b.data)
        c = result.materialize(value, "c")

        compiler = Compiler((result,))
        compiler.compile()
        slots = compiler.node_slots
        instruction = compiler.instructions[0]

        self.assertEqual(len(compiler.instructions), 1)
        self.assertIs(instruction.operation, operation.operation)
        self.assertEqual(
            instruction.input_slots, (slots[a.node], slots[b.node])
        )
        self.assertEqual(instruction.output_slot, slots[c.node])
        self.assertEqual(Computation(c).forward().tolist(), [5.0, 7.0, 9.0])

    def test_an_unmaterialized_operand_does_not_block_compilation(self):
        # Slots are numbered by vertex, so a pending operand compiles into
        # one like any other value. Only the runtime projection of that
        # program, and the Computation built from it, need the value itself.
        pending = VariableNode()
        leaf = ts.Variable([2.0], name="leaf")
        operation = OperationNode(Add())
        result = VariableNode()
        Edge(pending, operation, label="input_0")
        Edge(leaf.node, operation, label="input_1")
        Edge(operation, result, label="result")
        output = result.materialize([0.0], "output")

        compiler = Compiler((result,))
        instruction, = compiler.compile()

        self.assertIn(pending, compiler.node_slots)
        self.assertIn(compiler.node_slots[pending], compiler.leaf_slots)
        self.assertEqual(
            instruction.input_slots,
            (compiler.node_slots[pending], compiler.node_slots[leaf.node]),
        )
        self.assertEqual(
            compiler.output_slots, (compiler.node_slots[output.node],)
        )
        # The program is adopted structurally, so the Computation exists;
        # the value nothing ever materialized is missing only when it runs.
        with self.assertRaisesRegex(RuntimeError, "leaf slot"):
            Computation(output).forward()


class EagerRecordingLifecycleTests(unittest.TestCase):
    """Eager recording names a result before it materializes one."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_the_result_vertex_is_recorded_before_its_variable_exists(self):
        left = ts.Variable([2.0], name="left")

        product = left * 3.0
        producer = product.node.producer

        # Node ids are allocated in creation order: the vertex naming the
        # result exists before the operation vertex that produces into it,
        # and the Variable is materialized against it afterwards.
        self.assertLess(product.node.id, producer.id)
        self.assertTrue(product.node.is_bound)
        self.assertIs(product.node.variable, product)

    def test_every_recorded_vertex_is_materialized(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        output = ts.sum(ts.relu(x * y + 1.0))

        for node in Computation(output).nodes:
            if isinstance(node, VariableNode):
                with self.subTest(node=node):
                    self.assertTrue(node.is_bound)
                    self.assertIs(node.variable.node, node)


if __name__ == "__main__":
    unittest.main()
