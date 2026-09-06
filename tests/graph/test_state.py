import unittest
import gc
import weakref
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.compiler import Compiler
from tensors.graph.node import OperationNode, VariableNode
from tensors.graph.state import TraceScope, get_graph_state, reset_graph_state
from tensors.math.concat import Concat
from tensors.math.where import Where
from tensors.ops import Add


class GraphStateTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_state_connects_new_edges_to_their_source_and_target_nodes(self):
        # The registry records vertices, not values: a graph can be connected
        # before anything has been materialized into it.
        state = get_graph_state()
        source = state.add_variable_node()
        target = state.add_variable_node()
        edge = state.add_edge(source, target, label="value")

        self.assertEqual(state.nodes, [source, target])
        self.assertEqual(state.edges, [edge])
        self.assertEqual(source.outputs, [target])
        self.assertEqual(target.inputs, [source])

    def test_adding_an_edge_registers_external_nodes_with_the_state(self):
        first_state = get_graph_state()
        source = first_state.add_variable_node()
        reset_graph_state()
        state = get_graph_state()
        target = state.add_variable_node()

        state.add_edge(source, target)

        self.assertEqual(state.nodes, [target, source])

    def test_adding_edges_does_not_duplicate_registered_nodes(self):
        state = get_graph_state()
        source = state.add_variable_node()
        target = state.add_variable_node()

        state.add_edge(source, target)
        state.add_edge(source, target)

        self.assertEqual(state.nodes, [source, target])

    def test_outer_trace_scope_resets_state_and_nested_scope_reuses_it(self):
        previous = get_graph_state()
        previous.add_variable_node()

        outer = TraceScope()
        active = get_graph_state()
        inner = TraceScope()

        self.assertTrue(outer.outermost)
        self.assertFalse(inner.outermost)
        self.assertIsNot(active, previous)
        self.assertIs(get_graph_state(), active)

        inner.close()
        outer.close()

    def test_closing_a_trace_scope_is_idempotent(self):
        scope = TraceScope()

        scope.close()
        scope.close()
        next_scope = TraceScope()

        self.assertTrue(next_scope.outermost)
        next_scope.close()

    def test_state_does_not_retain_a_discarded_eager_computation(self):
        state = get_graph_state()
        value = ts.Variable([2.0])
        result = value * 3.0
        producer = result.node.producer
        references = [
            weakref.ref(result),
            weakref.ref(result.node),
            weakref.ref(producer),
            weakref.ref(producer.operands[1].node),
            weakref.ref(result.node._in_edges[0]),
            weakref.ref(producer._in_edges[0]),
        ]

        del result, producer
        gc.collect()

        # The Variable and VariableNode reference each other, so the whole
        # unreachable computation is a cycle. It must still be collectable.
        for index, reference in enumerate(references):
            with self.subTest(reference=index):
                self.assertIsNone(reference())
        self.assertEqual(state.nodes, [value.node])
        self.assertEqual(state.edges, [])
        self.assertEqual(value.node.outputs, [])

    def test_retained_result_keeps_the_upstream_graph_alive(self):
        state = get_graph_state()
        value = ts.Variable([2.0])
        result = value * 3.0 + 1.0
        scalar = result.node.producer.operands[1]
        scalar_reference = weakref.ref(scalar)
        product = result.node.producer.operands[0]
        product_reference = weakref.ref(product)

        del scalar, product
        state.clear()
        gc.collect()

        # A retained result must keep every Variable its replay and
        # differentiation need, even after the registry forgets them.
        self.assertIsNotNone(scalar_reference())
        self.assertIsNotNone(product_reference())
        self.assertEqual(Computation(result).forward().tolist(), [7.0])
        self.assertEqual(ts.grad(result, value).tolist(), [3.0])

    def test_clear_forgets_registrations_without_invalidating_graph(self):
        state = get_graph_state()
        value = ts.Variable([2.0])
        result = value * 3.0

        state.clear()

        self.assertEqual(state.nodes, [])
        self.assertEqual(state.edges, [])
        self.assertEqual(ts.grad(result, value).tolist(), [3.0])

    def test_eager_recording_resumes_after_a_graph_trace(self):
        @ts.Graph
        def model(value):
            return value * 2.0

        model(ts.Tensor([3.0]))
        eager = ts.Variable([4.0]) + 1.0
        state = get_graph_state()

        self.assertIn(eager.node, state.nodes)
        self.assertIn(eager.node._in_edges[0], state.edges)


class OperationRecordingTests(unittest.TestCase):
    """The graph layer owns how one operation invocation is assembled."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _operands(count):
        state = get_graph_state()
        return state, tuple(state.add_variable_node() for _ in range(count))

    def test_recording_an_operation_returns_an_unbound_result_vertex(self):
        state, operands = self._operands(2)

        result = state.record_operation(Add(), operands)

        self.assertIsInstance(result, VariableNode)
        self.assertFalse(result.is_bound)
        self.assertIn(result, state.nodes)

    def test_recording_an_operation_creates_one_operation_vertex(self):
        state, operands = self._operands(2)

        result = state.record_operation(Add(), operands)

        recorded = [
            node for node in state.nodes if isinstance(node, OperationNode)
        ]
        self.assertEqual(len(recorded), 1)
        self.assertIs(result.producer, recorded[0])
        self.assertIsInstance(recorded[0].operation, Add)

    def test_operand_edges_keep_their_order_and_labels(self):
        state, operands = self._operands(3)

        result = state.record_operation(Where(), operands)
        producer = result.producer

        self.assertEqual(producer.operand_nodes, operands)
        self.assertEqual(
            [edge.label for edge in producer._in_edges],
            ["input_0", "input_1", "input_2"],
        )

    def test_more_operands_than_named_labels_still_stay_ordered(self):
        state, operands = self._operands(7)

        result = state.record_operation(Concat(axis=0), operands)

        self.assertEqual(result.producer.operand_nodes, operands)
        self.assertEqual(
            [edge.label for edge in result.producer._in_edges],
            [f"input_{index}" for index in range(7)],
        )

    def test_the_result_edge_points_at_the_returned_vertex(self):
        state, operands = self._operands(2)

        result = state.record_operation(Add(), operands)
        outgoing = result.producer._out_edges

        self.assertEqual(len(outgoing), 1)
        self.assertIs(outgoing[0].target, result)
        self.assertEqual(outgoing[0].label, "result")
        self.assertIs(result.producer.result_node, result)

    def test_recording_an_operation_executes_nothing(self):
        state, operands = self._operands(2)
        calls = []
        original = Add.forward

        def counted(self, *args):
            calls.append(args)
            return original(self, *args)

        with patch.object(Add, "forward", counted):
            result = state.record_operation(Add(), operands)

        self.assertEqual(calls, [])
        self.assertFalse(result.is_bound)
        for operand in operands:
            self.assertFalse(operand.is_bound)

    def test_a_recorded_operation_compiles_without_any_runtime_value(self):
        # Structure is expressible on its own: nothing here holds a value,
        # and the fragment still compiles into an executable program.
        state, operands = self._operands(2)

        result = state.record_operation(Add(), operands)

        compiler = Compiler((result,), boundaries=operands)
        instruction, = compiler.compile()
        self.assertEqual(instruction.operation.name, "add")
        self.assertEqual(
            instruction.input_slots,
            tuple(compiler.node_slots[operand] for operand in operands),
        )
        self.assertEqual(instruction.output_slot, compiler.node_slots[result])
        self.assertFalse(result.is_bound)


if __name__ == "__main__":
    unittest.main()
