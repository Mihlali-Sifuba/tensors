import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation, UnboundVariableNodeError
from tensors.graph.computation.instruction import Instruction
from tensors.graph.edge import Edge
from tensors.graph.node import OperationNode, VariableNode
from tensors.graph.state import reset_graph_state
from tensors.math.sin import Sin
from tensors.ops import Add, Mul


class ComputationTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_computation_owns_backward_pass(self):
        value = ts.Variable([3.0])
        loss = ts.math.sum(value * value)

        ts.graph.Computation(loss).backward()

        self.assertEqual(value.grad.tolist(), [6.0])

    def test_computation_rejects_non_variable_output(self):
        with self.assertRaisesRegex(TypeError, "graph node"):
            ts.graph.Computation(ts.Tensor([1.0]))

    def test_computation_uses_explicit_gradient_seed(self):
        value = ts.Variable([2.0, 3.0])
        result = value * value

        ts.graph.Computation(result).backward(ts.Tensor([10.0, 20.0]))

        self.assertEqual(value.grad.tolist(), [40.0, 120.0])

    def test_computation_casts_gradient_seed_to_output_dtype(self):
        value = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        result = value * value

        ts.graph.Computation(result).backward(
            ts.Tensor([3.0], dtype=ts.float64)
        )

        self.assertIs(value.grad.dtype, ts.float32)
        self.assertEqual(value.grad.tolist(), [12.0])

    def test_computation_restores_input_gradient_dtypes_with_create_graph(self):
        left = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        right = ts.Variable(ts.Tensor([3.0], dtype=ts.float64))
        output = ts.sum(left * right)

        left_gradient, right_gradient = ts.grad(
            output,
            (left, right),
            create_graph=True,
        )
        cross_gradient = ts.grad(ts.sum(left_gradient), right)

        self.assertIs(left_gradient.dtype, ts.float32)
        self.assertIs(right_gradient.dtype, ts.float64)
        self.assertEqual(left_gradient.data.tolist(), [3.0])
        self.assertEqual(right_gradient.data.tolist(), [2.0])
        self.assertEqual(cross_gradient.tolist(), [1.0])

    def test_computation_rejects_gradient_shape_mismatch(self):
        value = ts.Variable([2.0, 3.0])
        result = value * value

        with self.assertRaisesRegex(ValueError, "Gradient shape"):
            ts.graph.Computation(result).backward(ts.Tensor([1.0]))

    def test_multi_output_graph_exposes_computations_tuple(self):
        @ts.Graph
        def model(x):
            return x + 1.0, x * 2.0

        outputs = model(ts.Tensor([3.0]))

        self.assertEqual(outputs[0].data.tolist(), [4.0])
        self.assertEqual(outputs[1].data.tolist(), [6.0])
        self.assertEqual(len(model.computations), 2)

    def test_multi_output_computations_share_one_execution_plan(self):
        @ts.Graph
        def model(x):
            trunk = x * 2.0
            return trunk + 1.0, trunk - 1.0

        model(ts.Tensor([3.0]))
        first, second = model.computations

        self.assertIs(first._instructions, second._instructions)
        self.assertIs(first._fusions, second._fusions)
        self.assertEqual(
            [node.label for node in first.nodes],
            ["var", "var", "mul", "var", "var", "add", "var"],
        )
        self.assertEqual(
            [node.label for node in second.nodes],
            ["var", "var", "mul", "var", "var", "sub", "var"],
        )

    def test_single_computation_property_rejects_multi_output_graph(self):
        @ts.Graph
        def model(x):
            return x + 1.0, x * 2.0

        model(ts.Tensor([3.0]))

        with self.assertRaisesRegex(RuntimeError, "multiple outputs"):
            _ = model.computation

    def test_external_loss_backpropagates_into_model_parameters(self):
        class Linear(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        model = Linear()
        loss = ts.math.sum(model(ts.Tensor([3.0])))

        ts.backward(loss)

        self.assertEqual(model.weight.grad.tolist(), [3.0])
        self.assertEqual(model.bias.grad.tolist(), [1.0])

    def test_computation_caches_its_dependency_order(self):
        value = ts.Variable([2.0])
        result = (value + 1.0) * 3.0
        computation = ts.graph.Computation(result)
        cached_order = computation._view_nodes

        first = computation.nodes
        second = computation.nodes
        first.clear()

        self.assertIs(computation._view_nodes, cached_order)
        self.assertEqual(second, list(cached_order))
        self.assertEqual(computation.nodes, list(cached_order))
        self.assertEqual(computation.forward().tolist(), [9.0])

    def test_released_computation_rejects_further_work(self):
        value = ts.Variable([2.0])
        result = value * 3.0
        computation = ts.graph.Computation(result)

        computation.release()
        computation.release()

        with self.assertRaisesRegex(RuntimeError, "released"):
            _ = computation.nodes
        with self.assertRaisesRegex(RuntimeError, "released"):
            computation.forward()
        with self.assertRaisesRegex(RuntimeError, "released"):
            computation.backward()


class ExecutionModelTests(unittest.TestCase):
    """The execution model is Computation plus ordered Instructions."""

    def test_instruction_holds_only_the_operation_and_its_slots(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        computation = Computation(x * y)
        instruction = computation._instructions[0]

        self.assertIsInstance(instruction, Instruction)
        self.assertEqual(
            Instruction.__slots__,
            ("operation", "input_slots", "output_slot"),
        )
        self.assertIsInstance(instruction.operation, ts.ops.Operation)
        self.assertEqual(
            instruction.input_slots,
            (
                computation._node_slots[x.node],
                computation._node_slots[y.node],
            ),
        )
        self.assertEqual(
            computation._variable_nodes[instruction.output_slot].variable,
            computation.output,
        )

    def test_instruction_is_immutable(self):
        computation = Computation(ts.Variable([2.0]) * ts.Variable([3.0]))
        instruction = computation._instructions[0]

        with self.assertRaises(AttributeError):
            instruction.output_slot = 0
        with self.assertRaises(AttributeError):
            instruction.operation = None

    def test_removed_execution_containers_are_gone(self):
        from tensors.graph import computation as module

        for name in (
            "_ForwardInstruction",
            "_ReverseDemand",
            "_ExecutionWorkspace",
            "_ForwardGroup",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(module, name))

    def test_fusion_is_metadata_beside_the_instruction_sequence(self):
        value = ts.Variable(ts.full((4_096,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)
        computation = Computation(output)

        self.assertEqual(
            [instruction.operation.name
             for instruction in computation._instructions],
            ["mul", "sin", "add", "sum"],
        )
        # The three elementwise instructions form one fusible run; the
        # reduction stays ordinary and carries no metadata.
        self.assertEqual(list(computation._fusions), [0])
        end, steps, source_slots = computation._fusions[0]
        self.assertEqual(end, 2)
        self.assertEqual([step[0] for step in steps], ["multiply", "sin", "add"])
        self.assertEqual(computation._fusion_starts, {2: 0})
        self.assertNotIn(3, computation._fusions)

        # Instruction itself carries no fusion state.
        for name in ("fused", "fusion_id", "fused_steps", "shape", "dtype"):
            with self.subTest(field=name):
                self.assertNotIn(name, Instruction.__slots__)

    def test_instruction_sequence_is_identical_across_backends(self):
        def describe(backend: str) -> list[tuple[str, tuple[int, ...], int]]:
            with ts.use_backend(backend):
                value = ts.Variable(ts.full((64,), 0.5))
                output = ts.sum(ts.sin(value * 1.5) + 0.25)
                return [
                    (
                        instruction.operation.name,
                        instruction.input_slots,
                        instruction.output_slot,
                    )
                    for instruction in Computation(output)._instructions
                ]

        reference = describe("python")
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                self.assertEqual(describe(backend), reference)

    def test_concurrent_replay_does_not_share_execution_buffers(self):
        import threading

        value = ts.Variable(ts.full((64,), 2.0))
        output = value * 3.0 + 1.0
        computation = Computation(output)
        results: list[float] = []
        failures: list[BaseException] = []
        barrier = threading.Barrier(4)

        def replay() -> None:
            try:
                barrier.wait()
                for _ in range(50):
                    results.append(computation.forward()[0])
            except BaseException as error:  # pragma: no cover - reported below
                failures.append(error)

        workers = [threading.Thread(target=replay) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertEqual(failures, [])
        self.assertEqual(set(results), {7.0})


class GraphFirstExecutionTests(unittest.TestCase):
    """A compiled graph executes into slots and materializes what it names."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _pending_program():
        """Return the vertices of ``c = a + b`` and ``d = c * a``.

        Only the leaves hold values: the graph names its results before
        anything has produced them.
        """
        a = ts.Variable([1.0, 2.0, 3.0], name="a")
        b = ts.Variable([4.0, 5.0, 6.0], name="b")
        addition = OperationNode(Add())
        product = OperationNode(Mul())
        c = VariableNode()
        d = VariableNode()
        Edge(a.node, addition, label="input_0")
        Edge(b.node, addition, label="input_1")
        Edge(addition, c, label="result")
        Edge(c, product, label="input_0")
        Edge(a.node, product, label="input_1")
        Edge(product, d, label="result")
        return a, b, c, d

    def test_a_computation_is_built_from_vertices_that_hold_no_value(self):
        a, b, c, d = self._pending_program()

        computation, = Computation.from_nodes((d,))

        self.assertFalse(c.is_bound)
        self.assertFalse(d.is_bound)
        self.assertEqual(computation._variable_nodes, (a.node, b.node, c, d))
        self.assertEqual(computation._output_slot, computation._node_slots[d])
        self.assertEqual(
            [i.operation.name for i in computation._instructions],
            ["add", "mul"],
        )

    def test_construction_materializes_nothing(self):
        _, _, c, d = self._pending_program()

        computation, = Computation.from_nodes((d,))

        self.assertIsNone(computation._variables)
        self.assertFalse(c.is_bound or d.is_bound)
        with self.assertRaises(UnboundVariableNodeError):
            computation.output
        with self.assertRaisesRegex(RuntimeError, "Run forward"):
            computation.backward()

    def test_forward_seeds_leaves_and_materializes_every_result(self):
        a, b, c, d = self._pending_program()
        computation, = Computation.from_nodes((d,))

        result = computation.forward()

        self.assertEqual(result.tolist(), [5.0, 14.0, 27.0])
        self.assertTrue(c.is_bound and d.is_bound)
        self.assertEqual(c.variable.data.tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(d.variable.data.tolist(), [5.0, 14.0, 27.0])
        # Each materialized Variable belongs to the vertex that named it.
        self.assertIs(c.variable.node, c)
        self.assertIs(d.variable.node, d)
        self.assertIs(computation.output, d.variable)
        self.assertIs(computation.output.data, result)

    def test_a_materialized_result_follows_its_operands_gradient_demand(self):
        a, _, c, d = self._pending_program()
        Computation.from_nodes((d,))[0].forward()
        self.assertTrue(c.variable.requires_grad)

        reset_graph_state()
        left = ts.Variable([2.0], name="left", requires_grad=False)
        right = ts.Variable([3.0], name="right", requires_grad=False)
        operation = OperationNode(Add())
        total = VariableNode()
        Edge(left.node, operation, label="input_0")
        Edge(right.node, operation, label="input_1")
        Edge(operation, total, label="result")

        Computation.from_nodes((total,))[0].forward()

        self.assertFalse(total.variable.requires_grad)
        self.assertEqual(total.variable.data.tolist(), [5.0])

    def test_replay_updates_the_variable_a_vertex_already_names(self):
        a, _, _, d = self._pending_program()
        computation, = Computation.from_nodes((d,))
        computation.forward()
        materialized = d.variable

        a.data = ts.Tensor([2.0, 2.0, 2.0])
        replayed = computation.forward()

        # A second pass updates the Variable the vertex names rather than
        # binding another one to it.
        self.assertIs(d.variable, materialized)
        self.assertIs(computation.output, materialized)
        self.assertEqual(replayed.tolist(), [12.0, 14.0, 16.0])
        self.assertEqual(materialized.data.tolist(), [12.0, 14.0, 16.0])

    def test_results_materialize_in_dependency_order(self):
        _, _, c, d = self._pending_program()
        computation, = Computation.from_nodes((d,))
        observed = []
        original = Mul.forward

        def watched(self, *args):
            observed.append((c.is_bound, d.is_bound))
            return original(self, *args)

        with patch.object(Mul, "forward", watched):
            computation.forward()

        # The product runs after the sum has given its vertex a value and
        # before its own vertex has one.
        self.assertEqual(observed, [(True, False)])

    def test_an_unbound_leaf_is_reported_as_a_missing_value(self):
        pending = VariableNode()
        leaf = ts.Variable([2.0], name="leaf")
        operation = OperationNode(Add())
        result = VariableNode()
        Edge(pending, operation, label="input_0")
        Edge(leaf.node, operation, label="input_1")
        Edge(operation, result, label="result")
        computation, = Computation.from_nodes((result,))

        with self.assertRaisesRegex(RuntimeError, "leaf slot .* no value"):
            computation.forward()
        self.assertFalse(result.is_bound)

    def test_each_view_produces_its_own_output(self):
        a, b, c, d = self._pending_program()

        first, second = Computation.from_nodes((c, d))

        self.assertIs(first._variable_nodes, second._variable_nodes)
        self.assertEqual(
            [i.operation.name for i in first._view_instructions], ["add"]
        )
        self.assertEqual(
            [i.operation.name for i in second._view_instructions],
            ["add", "mul"],
        )
        self.assertEqual(first.forward().tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(second.forward().tolist(), [5.0, 14.0, 27.0])
        self.assertIs(first.output, c.variable)
        self.assertIs(second.output, d.variable)

    def test_a_graph_first_program_differentiates_after_it_has_run(self):
        a, b, _, d = self._pending_program()
        computation, = Computation.from_nodes((d,))
        computation.forward()

        computation.backward(ts.Tensor([1.0, 1.0, 1.0]))

        # d = (a + b) * a, so dd/da = (a + b) + a and dd/db = a.
        self.assertEqual(a.grad.tolist(), [6.0, 9.0, 12.0])
        self.assertEqual(b.grad.tolist(), [1.0, 2.0, 3.0])

    def test_adopting_a_program_resolves_no_runtime_values(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * 3.0)

        computation = Computation(output)

        # Construction is structural even for a fully materialized graph:
        # the Variables behind the slots are resolved by the pass that needs
        # them, which for this program is differentiation.
        self.assertIsNone(computation._variables)
        computation.forward()
        computation.backward()
        self.assertEqual(
            computation._variables,
            tuple(node.variable for node in computation._variable_nodes),
        )

    def test_fusion_is_planned_once_the_program_holds_values(self):
        value = ts.Variable(ts.full((4_096,), 0.5), name="value")
        first_operation = OperationNode(Sin())
        second_operation = OperationNode(Sin())
        middle = VariableNode()
        end = VariableNode()
        Edge(value.node, first_operation, label="input_0")
        Edge(first_operation, middle, label="result")
        Edge(middle, second_operation, label="input_0")
        Edge(second_operation, end, label="result")
        computation, = Computation.from_nodes((end,))

        # A fusible run is recognized from the shapes and dtypes its slots
        # hold, so there is nothing to plan before the program has run.
        self.assertEqual(computation._fusions, {})
        expected = computation.forward().tolist()

        self.assertEqual(list(computation._fusions), [0])
        self.assertEqual(computation._fusion_starts, {1: 0})
        self.assertEqual(computation.forward().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
