import unittest

from tensors.graph.state import TraceScope, get_graph_state, reset_graph_state


class GraphStateTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_state_connects_new_edges_to_their_source_and_target_nodes(self):
        state = get_graph_state()
        source = state.add_node("source")
        target = state.add_node("target")
        edge = state.add_edge(source, target, label="value")

        self.assertEqual(state.nodes, [source, target])
        self.assertEqual(state.edges, [edge])
        self.assertEqual(source.outputs, [target])
        self.assertEqual(target.inputs, [source])

    def test_adding_an_edge_registers_external_nodes_with_the_state(self):
        first_state = get_graph_state()
        source = first_state.add_node("source")
        reset_graph_state()
        state = get_graph_state()
        target = state.add_node("target")

        state.add_edge(source, target)

        self.assertEqual(state.nodes, [target, source])

    def test_outer_trace_scope_resets_state_and_nested_scope_reuses_it(self):
        previous = get_graph_state()
        previous.add_node("stale")

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


if __name__ == "__main__":
    unittest.main()
