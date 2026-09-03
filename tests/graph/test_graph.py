import unittest
import gc
import threading
import weakref

import tensors as ts
from tensors.graph.state import reset_graph_state


class GraphTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_graph_package_owns_structural_types(self):
        self.assertEqual(ts.graph.Computation.__module__, "tensors.graph.computation")
        self.assertEqual(ts.graph.Edge.__module__, "tensors.graph.edge")
        self.assertEqual(ts.graph.Node.__module__, "tensors.graph.node")

    def test_subclass_traces_fresh_with_tensor_inputs(self):
        class Linear(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        model = Linear()
        first = model(ts.Tensor([3.0]))
        self.assertEqual(first.data.tolist(), [7.0])
        self.assertIs(model.computation.output, first)
        second = model(ts.Tensor([4.0]))

        self.assertIsNot(first, second)
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertIs(model.computation.output, second)
        # input, weight, mul, product, bias, add, result
        self.assertEqual(len(model.nodes), 7)
        self.assertEqual(model.parameters(), [model.weight, model.bias])

    def test_function_graph_works_as_a_decorator_target(self):
        weight = ts.Variable([3.0])

        @ts.Graph
        def model(x):
            return x * weight

        result = model(ts.Tensor([2.0]))

        self.assertEqual(result.data.tolist(), [6.0])
        self.assertEqual(model.parameters(), [weight])

    def test_graph_traces_fresh_keyword_inputs(self):
        @ts.Graph
        def model(x, scale):
            return x * scale

        first = model(ts.Tensor([2.0]), scale=ts.Tensor([3.0]))
        second = model(ts.Tensor([4.0]), scale=ts.Tensor([5.0]))

        self.assertIsNot(first, second)
        self.assertEqual(first.data.tolist(), [6.0])
        self.assertEqual(second.data.tolist(), [20.0])
        self.assertIs(model.computation.output, second)

    def test_graph_preserves_boolean_configuration_arguments(self):
        @ts.Graph
        def model(value, *, add_bias=False):
            return value + 1.0 if add_bias else value * 1.0

        without_bias = model(ts.Tensor([2.0]), add_bias=False)
        with_bias = model(ts.Tensor([2.0]), add_bias=True)

        self.assertEqual(without_bias.data.tolist(), [2.0])
        self.assertEqual(with_bias.data.tolist(), [3.0])

    def test_graph_preserves_axis_configuration_arguments(self):
        @ts.Graph
        def model(value, *, axis=-1):
            return ts.softmax(value, axis=axis)

        value = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])

        result = model(value, axis=0)
        expected = ts.softmax(value, axis=0)

        self.assertEqual(result.data.tolist(), expected.tolist())

    def test_graph_preserves_scalar_configuration_arguments(self):
        @ts.Graph
        def model(value, *, scale=1.0):
            return value * scale

        result = model(ts.Tensor([2.0]), scale=3.0)
        operands = result.node.producer.operands

        self.assertEqual(result.data.tolist(), [6.0])
        self.assertEqual(operands[1].data.item(), 3.0)
        self.assertFalse(operands[1].requires_grad)

    def test_graph_uses_python_keyword_errors_on_fresh_trace(self):
        @ts.Graph
        def model(x, scale):
            return x * scale

        model(ts.Tensor([2.0]), scale=ts.Tensor([3.0]))

        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            model(ts.Tensor([2.0]), other=ts.Tensor([3.0]))

    def test_graph_uses_python_input_count_errors_on_fresh_trace(self):
        @ts.Graph
        def model(x):
            return x * 2.0

        model(ts.Tensor([2.0]))

        with self.assertRaisesRegex(TypeError, "takes 1 positional argument"):
            model(ts.Tensor([2.0]), ts.Tensor([3.0]))

    def test_graph_rebuild_traces_new_input_shape(self):
        @ts.Graph
        def model(x):
            return x * 2.0

        model(ts.Tensor([1.0]))
        rebuilt = model.rebuild(ts.Tensor([1.0, 2.0]))

        self.assertEqual(rebuilt.shape, (2,))
        self.assertEqual(rebuilt.data.tolist(), [2.0, 4.0])

    def test_graph_rejects_non_variable_output(self):
        class BadGraph(ts.Graph):
            def forward(self, x):
                return x.data

        with self.assertRaisesRegex(TypeError, "must return"):
            BadGraph()(ts.Tensor([1.0]))

    def test_graph_collects_parameters_from_child_graphs_and_containers(self):
        class Child(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, x):
                return x * self.weight

        class Parent(ts.Graph):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.extra = [ts.Variable([1.0])]
                self.frozen = ts.Variable([0.0], requires_grad=False)

            def forward(self, x):
                return self.child(x) + self.extra[0]

        model = Parent()

        self.assertEqual(model.parameters(), [model.child.weight, model.extra[0]])

    def test_graph_collects_private_trainable_attributes(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self._weight = ts.Variable([2.0])

            def forward(self, value):
                return value * self._weight

        model = Scale()

        self.assertEqual(model.parameters(), [model._weight])

    def test_parameter_cache_tracks_attributes_and_gradient_flags(self):
        class Model(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, value):
                return value * self.weight

        model = Model()
        self.assertEqual(model.parameters(), [model.weight])
        self.assertIsNotNone(model._parameter_cache)

        model.weight.requires_grad = False
        self.assertEqual(model.parameters(), [])
        model.bias = ts.Variable([1.0])
        self.assertEqual(model.parameters(), [model.bias])

    def test_mutable_parameter_containers_remain_dynamically_discovered(self):
        class Model(ts.Graph):
            def __init__(self):
                super().__init__()
                self.layers = [ts.Variable([2.0])]

            def forward(self, value):
                return value * self.layers[0]

        model = Model()
        self.assertEqual(model.parameters(), [model.layers[0]])
        self.assertIsNone(model._parameter_cache)

        model.layers.append(ts.Variable([3.0]))
        self.assertEqual(model.parameters(), model.layers)

    def test_graph_allows_input_shape_changes_on_fresh_trace(self):
        @ts.Graph
        def model(x):
            return x * 2.0

        first = model(ts.Tensor([1.0]))
        second = model(ts.Tensor([1.0, 2.0]))

        self.assertEqual(first.shape, (1,))
        self.assertEqual(first.data.tolist(), [2.0])
        self.assertEqual(second.shape, (2,))
        self.assertEqual(second.data.tolist(), [2.0, 4.0])
        self.assertIs(model.computation.output, second)

    def test_compile_replays_matching_tensor_inputs_without_retracing(self):
        calls = 0

        @ts.Graph
        def model(x):
            nonlocal calls
            calls += 1
            return x * 2.0 + 1.0

        first = model.compile(ts.Tensor([2.0]))
        second = model(ts.Tensor([4.0]))

        self.assertIs(first, second)
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertEqual(calls, 1)

    def test_compile_guard_retraces_shape_and_static_argument_changes(self):
        calls = 0

        @ts.Graph
        def model(x, *, scale=1.0):
            nonlocal calls
            calls += 1
            return x * scale

        model.compile(ts.Tensor([2.0]), scale=2.0)
        same = model(ts.Tensor([3.0]), scale=2.0)
        resized = model(ts.Tensor([3.0, 4.0]), scale=2.0)
        reconfigured = model(ts.Tensor([3.0, 4.0]), scale=3.0)

        self.assertEqual(same.data.tolist(), [6.0])
        self.assertEqual(resized.data.tolist(), [6.0, 8.0])
        self.assertEqual(reconfigured.data.tolist(), [9.0, 12.0])
        self.assertEqual(calls, 3)

    def test_compile_rejects_variable_inputs_and_uncompile_restores_tracing(self):
        calls = 0

        @ts.Graph
        def model(x):
            nonlocal calls
            calls += 1
            return x + 1.0

        with self.assertRaisesRegex(TypeError, "Variable inputs"):
            model.compile(ts.Variable([1.0]))

        model.compile(ts.Tensor([1.0]))
        model(ts.Tensor([2.0]))
        model.uncompile()
        model(ts.Tensor([3.0]))

        self.assertEqual(calls, 2)

    def test_rebuild_bypasses_a_matching_compiled_trace(self):
        calls = 0

        @ts.Graph
        def model(x):
            nonlocal calls
            calls += 1
            return x * 2.0

        model.compile(ts.Tensor([1.0]))
        rebuilt = model.rebuild(ts.Tensor([3.0]))

        self.assertEqual(rebuilt.data.tolist(), [6.0])
        self.assertEqual(calls, 2)

    def test_compiled_trace_rebuilds_after_parameter_replacement(self):
        class Model(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                return value * self.weight

        model = Model()
        model.compile(ts.Tensor([3.0]))
        model.weight = ts.Variable([4.0])
        result = model(ts.Tensor([3.0]))

        self.assertEqual(result.data.tolist(), [12.0])
        self.assertEqual(model.calls, 2)

    def test_compiled_trace_rebuilds_after_its_computation_is_released(self):
        calls = 0

        @ts.Graph
        def model(value):
            nonlocal calls
            calls += 1
            return value + 1.0

        model.compile(ts.Tensor([1.0]))
        model.computation.release()
        result = model(ts.Tensor([2.0]))

        self.assertEqual(result.data.tolist(), [3.0])
        self.assertEqual(calls, 2)

    def test_new_call_does_not_retain_the_previous_computation(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, value):
                return value * self.weight

        model = Scale()
        first = model(ts.Tensor([3.0]))
        first_reference = weakref.ref(first)
        first_node_reference = weakref.ref(first.node)

        del first
        model(ts.Tensor([4.0]))
        gc.collect()

        self.assertIsNone(first_reference())
        self.assertIsNone(first_node_reference())

    def test_release_drops_graph_references_but_not_returned_output(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, value):
                return value * self.weight

        model = Scale()
        result = model(ts.Tensor([3.0]))

        model.release()

        self.assertEqual(model.nodes, [])
        self.assertEqual(model.edges, [])
        with self.assertRaisesRegex(RuntimeError, "has not been called"):
            _ = model.computation
        self.assertEqual(ts.grad(result, model.weight).tolist(), [3.0])
        self.assertEqual(model(ts.Tensor([4.0])).data.tolist(), [8.0])

    def test_same_graph_keeps_latest_computation_per_thread(self):
        class Scale(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])

            def forward(self, value):
                return value * self.weight

        model = Scale()
        barrier = threading.Barrier(2)
        results = {}
        errors = []
        results_lock = threading.Lock()

        def run(name, value):
            try:
                output = model(ts.Tensor([value]))
                barrier.wait(timeout=5)
                with results_lock:
                    results[name] = (output, model.computation.output)
            except BaseException as exc:
                with results_lock:
                    errors.append(exc)

        workers = [
            threading.Thread(target=run, args=("first", 3.0)),
            threading.Thread(target=run, args=("second", 4.0)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)

        self.assertFalse(any(worker.is_alive() for worker in workers))
        if errors:
            self.fail(f"threaded graph execution failed: {errors!r}")
        self.assertEqual(set(results), {"first", "second"})
        for output, recorded_output in results.values():
            self.assertIs(output, recorded_output)


if __name__ == "__main__":
    unittest.main()
