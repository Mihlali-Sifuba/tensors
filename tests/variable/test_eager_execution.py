import ast
import gc
import inspect
import pathlib
import unittest
from unittest.mock import patch

import tensors as ts
from tensors import variable as variable_module
from tensors.graph.computation.compiler import Compiler
from tensors.graph.state import reset_graph_state
from tensors.ops import Add, Mul, Neg, Operation


class EagerOperationLifecycleTests(unittest.TestCase):
    """An eager operation is compiled and executed like any other."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_the_result_vertex_exists_and_is_empty_while_the_operation_runs(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")
        observed = {}
        original = Add.forward

        def watched(self, *args):
            # The fragment is already wired: the operands reach an operation
            # vertex whose result vertex names a value nothing holds yet.
            operation_node = left.node.outputs[-1]
            result_node = operation_node.outputs[0]
            observed["operation"] = operation_node
            observed["result"] = result_node
            observed["bound"] = result_node.is_bound
            observed["operands"] = operation_node.operand_nodes
            return original(self, *args)

        with patch.object(Add, "forward", watched):
            total = left + right

        self.assertFalse(observed["bound"])
        self.assertEqual(observed["operands"], (left.node, right.node))
        # Execution materialized the Variable against the vertex that was
        # constructed before the operation ran.
        self.assertIs(total.node, observed["result"])
        self.assertIs(total.node.variable, total)
        self.assertIs(total.node.producer, observed["operation"])
        self.assertEqual(total.data.tolist(), [4.0, 6.0])

    def test_a_unary_operation_takes_the_same_path(self):
        value = ts.Variable([-2.0], name="value")
        observed = {}
        original = Neg.forward

        def watched(self, *args):
            result_node = value.node.outputs[-1].outputs[0]
            observed["bound"] = result_node.is_bound
            return original(self, *args)

        with patch.object(Neg, "forward", watched):
            negated = -value

        self.assertFalse(observed["bound"])
        self.assertEqual(negated.data.tolist(), [2.0])
        self.assertIs(negated.node.producer.operand_nodes[0], value.node)
        self.assertEqual(negated.node.producer.label, "neg")

    def test_scalar_and_tensor_operands_are_normalized_into_the_graph(self):
        value = ts.Variable([2.0], name="value")

        scaled = value * 3.0
        shifted = value + ts.Tensor([1.0])

        self.assertEqual(scaled.data.tolist(), [6.0])
        self.assertEqual(shifted.data.tolist(), [3.0])
        operands = scaled.node.producer.operand_nodes
        self.assertIs(operands[0], value.node)
        self.assertEqual(operands[1].variable.data.item(), 3.0)
        self.assertFalse(operands[1].variable.requires_grad)
        # Promotion is still resolved while the operand is built, not read
        # back off a result the graph has not calculated yet.
        integer = ts.Variable(
            ts.Tensor([2], dtype=ts.int32), requires_grad=False
        )
        self.assertEqual((integer * 3).dtype, ts.int32)
        self.assertEqual(value.astype("float32").dtype, ts.float32)


class EagerFragmentBoundaryTests(unittest.TestCase):
    """A new operation executes alone; its history stays in the graph."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_chained_operation_does_not_re_execute_earlier_ones(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")
        calls = []
        original = Add.forward

        def counted(self, *args):
            calls.append(args)
            return original(self, *args)

        with patch.object(Add, "forward", counted):
            total = left + right
            product = total * right

        self.assertEqual(len(calls), 1)
        self.assertEqual(total.data.tolist(), [4.0, 6.0])
        self.assertEqual(product.data.tolist(), [12.0, 24.0])

    def test_a_fragment_compiles_only_its_own_operation(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")
        total = left + right
        compiled = []
        original = Compiler.compile

        def counted(self):
            instructions = original(self)
            compiled.append(self)
            return instructions

        with patch.object(Compiler, "compile", counted):
            product = total * right

        compiler, = compiled
        self.assertEqual(
            [i.operation.name for i in compiler.instructions], ["mul"]
        )
        # The operands already hold their values, so they are the leaves of
        # this program and the graph behind them is never traversed.
        self.assertEqual(
            compiler.variable_nodes, (total.node, right.node, product.node)
        )
        self.assertEqual(sorted(compiler.leaf_slots), [0, 1])
        self.assertNotIn(left.node, compiler.nodes)
        self.assertIn(total.node, compiler.boundary_nodes)

    def test_the_persistent_graph_keeps_every_recorded_operation(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")

        total = left + right
        product = total * right

        self.assertEqual(product.node.producer.label, "mul")
        self.assertEqual(
            product.node.producer.operand_nodes, (total.node, right.node)
        )
        self.assertEqual(total.node.producer.label, "add")
        self.assertEqual(
            total.node.producer.operand_nodes, (left.node, right.node)
        )
        # A computation rooted at the later result still spans both.
        self.assertEqual(
            [
                node.label
                for node in ts.graph.Computation(product).nodes
                if node.label in {"add", "mul"}
            ],
            ["add", "mul"],
        )

    def test_backward_through_chained_operations_reaches_the_leaves(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")

        product = (left + right) * right
        ts.backward(ts.sum(product))

        # d = (left + right) * right, so dd/dleft = right and
        # dd/dright = (left + right) + right.
        self.assertEqual(left.grad.tolist(), [3.0, 4.0])
        self.assertEqual(right.grad.tolist(), [7.0, 10.0])


class EagerFailureTests(unittest.TestCase):
    """A failed operation raises; nothing executes it a second way."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_failing_operation_is_not_executed_a_second_way(self):
        attempts = []

        class Failing(Operation):
            name = "failing"

            def forward(self, value):
                attempts.append(value)
                raise ValueError("deliberate failure")

            def backward(self, gradient, value, *, needs_input_grad):
                return [gradient]

        value = ts.Variable([2.0], name="value")

        with self.assertRaisesRegex(ValueError, "deliberate failure"):
            ts.Variable._apply_operation(Failing(), (value,))

        # The operation ran once and nothing fell back to running it again.
        self.assertEqual(len(attempts), 1)
        self.assertEqual(value.data.tolist(), [2.0])

    def test_a_failed_operation_materializes_no_result(self):
        class Failing(Operation):
            name = "failing"

            def forward(self, value):
                raise ValueError("deliberate failure")

            def backward(self, gradient, value, *, needs_input_grad):
                return [gradient]

        value = ts.Variable([2.0], name="value")

        with self.assertRaises(ValueError):
            ts.Variable._apply_operation(Failing(), (value,))

        # Nothing holds the fragment recorded for the failed operation, and
        # outgoing edges are weak, so ordinary collection reclaims it.
        gc.collect()
        self.assertEqual(value.node.outputs, [])
        self.assertTrue(value.node.is_bound)


class EagerExecutionPathTests(unittest.TestCase):
    """Eager operators construct and delegate; they never execute."""

    def test_variable_never_executes_an_operation_itself(self):
        source = inspect.getsource(variable_module)
        executions = [
            node.lineno
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "forward"
            and node.args
        ]

        # Running the compiled fragment takes no operands. An operation
        # executed here would be handed values, and none is.
        self.assertEqual(executions, [])
        self.assertNotIn("_record_operation", source)
        self.assertTrue(hasattr(ts.Variable, "_apply_operation"))
        self.assertFalse(hasattr(ts.Variable, "_record_operation"))

    def test_no_eager_path_feeds_a_variable_value_to_an_operation(self):
        # Handing a Variable's Tensor to ``operation.forward`` is exactly
        # what eager execution used to do. Every such path now records a
        # graph fragment and executes it through a Computation instead.
        offenders = []
        root = pathlib.Path(inspect.getsourcefile(ts)).parent
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if not (
                    isinstance(function, ast.Attribute)
                    and function.attr == "forward"
                ):
                    continue
                for argument in node.args:
                    for inner in ast.walk(argument):
                        if (
                            isinstance(inner, ast.Attribute)
                            and inner.attr == "data"
                        ):
                            offenders.append(
                                f"{path.name}:{node.lineno}"
                            )
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
