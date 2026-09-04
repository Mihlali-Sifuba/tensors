import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.instruction import Instruction
from tensors.graph.state import reset_graph_state


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
                computation._variable_slots[x],
                computation._variable_slots[y],
            ),
        )
        self.assertEqual(
            computation._variables[instruction.output_slot],
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


if __name__ == "__main__":
    unittest.main()
