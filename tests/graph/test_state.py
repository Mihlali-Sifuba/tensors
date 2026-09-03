import unittest
import gc
import weakref

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import TraceScope, get_graph_state, reset_graph_state


class GraphStateTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_state_connects_new_edges_to_their_source_and_target_nodes(self):
        # Constructing a Variable registers its node, so start from a clean
        # state and register the vertices under test explicitly.
        variables = [ts.Variable([1.0]), ts.Variable([1.0])]
        reset_graph_state()
        state = get_graph_state()
        source = state.add_variable_node(variables[0])
        target = state.add_variable_node(variables[1])
        edge = state.add_edge(source, target, label="value")

        self.assertEqual(state.nodes, [source, target])
        self.assertEqual(state.edges, [edge])
        self.assertEqual(source.outputs, [target])
        self.assertEqual(target.inputs, [source])

    def test_adding_an_edge_registers_external_nodes_with_the_state(self):
        variables = [ts.Variable([1.0]), ts.Variable([1.0])]
        reset_graph_state()
        first_state = get_graph_state()
        source = first_state.add_variable_node(variables[0])
        reset_graph_state()
        state = get_graph_state()
        target = state.add_variable_node(variables[1])

        state.add_edge(source, target)

        self.assertEqual(state.nodes, [target, source])

    def test_adding_edges_does_not_duplicate_registered_nodes(self):
        variables = [ts.Variable([1.0]), ts.Variable([1.0])]
        reset_graph_state()
        state = get_graph_state()
        source = state.add_variable_node(variables[0])
        target = state.add_variable_node(variables[1])

        state.add_edge(source, target)
        state.add_edge(source, target)

        self.assertEqual(state.nodes, [source, target])

    def test_outer_trace_scope_resets_state_and_nested_scope_reuses_it(self):
        previous = get_graph_state()
        previous.add_variable_node(ts.Variable([1.0], name="stale"))

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


if __name__ == "__main__":
    unittest.main()
