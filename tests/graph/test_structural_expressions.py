import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.compiler import Compiler
from tensors.graph.expression import (
    UnsupportedStructuralExpression, as_tensor_operand,
)
from tensors.graph.node import OperationNode, VariableNode
from tensors.graph.state import (
    GraphState, get_graph_state, reset_graph_state,
)
from tensors.linalg.dot import Dot
from tensors.math.relu import ReLU
from tensors.ops import Add, Mul


class StructuralExpressionTests(unittest.TestCase):
    """An expression over vertices records a graph and calculates nothing."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_adding_two_vertices_records_an_addition(self):
        left = VariableNode()
        right = VariableNode()
        calls = []
        original = Add.forward

        def counted(self, *args):
            calls.append(args)
            return original(self, *args)

        with patch.object(Add, "forward", counted):
            total = left + right

        self.assertIsInstance(total, VariableNode)
        self.assertFalse(total.is_bound)
        self.assertEqual(calls, [])
        self.assertIsInstance(total.producer, OperationNode)
        self.assertIsInstance(total.producer.operation, Add)
        self.assertEqual(total.producer.operand_nodes, (left, right))
        self.assertIn(total, get_graph_state().nodes)

    def test_a_variable_operand_takes_part_as_its_vertex(self):
        node = VariableNode()
        variable = ts.Variable([1.0, 2.0], name="parameter")
        reads = []
        original = ts.Variable.data

        def watched(self):
            reads.append(self)
            return original.fget(self)

        with patch.object(ts.Variable, "data", property(watched)):
            result = node + variable

        # The parameter entered the graph as the vertex naming it; nothing
        # read the value it holds.
        self.assertEqual(reads, [])
        self.assertFalse(result.is_bound)
        self.assertEqual(result.producer.operand_nodes, (node, variable.node))

    def test_a_variable_on_the_left_records_the_same_way(self):
        node = VariableNode()
        variable = ts.Variable([1.0, 2.0], name="parameter")

        result = variable * node

        self.assertIsInstance(result, VariableNode)
        self.assertFalse(result.is_bound)
        self.assertIsInstance(result.producer.operation, Mul)
        # Operand order follows the expression, not the operand kinds.
        self.assertEqual(result.producer.operand_nodes, (variable.node, node))

    def test_matmul_with_a_parameter_records_a_dot(self):
        inputs = VariableNode()
        weight = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")

        product = inputs @ weight

        self.assertFalse(product.is_bound)
        self.assertIsInstance(product.producer.operation, Dot)
        self.assertEqual(
            product.producer.operand_nodes, (inputs, weight.node)
        )
        # The public function records the same operation.
        through_function = ts.matmul(inputs, weight)
        self.assertFalse(through_function.is_bound)
        self.assertIsInstance(through_function.producer.operation, Dot)

    def test_relu_records_structurally(self):
        node = VariableNode()

        activated = ts.relu(node)

        self.assertIsInstance(activated, VariableNode)
        self.assertFalse(activated.is_bound)
        self.assertIsInstance(activated.producer.operation, ReLU)
        self.assertEqual(activated.producer.operand_nodes, (node,))

    def test_a_chain_records_the_whole_topology_unbound(self):
        inputs = VariableNode()
        weight = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")
        bias = ts.Variable([0.5, 0.5], name="bias")

        hidden = inputs @ weight
        output = ts.relu(hidden + bias)

        self.assertFalse(inputs.is_bound)
        self.assertFalse(hidden.is_bound)
        self.assertFalse(output.is_bound)
        self.assertIsInstance(output.producer.operation, ReLU)
        total = output.producer.operand_nodes[0]
        self.assertIsInstance(total.producer.operation, Add)
        self.assertEqual(total.producer.operand_nodes, (hidden, bias.node))
        self.assertEqual(
            hidden.producer.operand_nodes, (inputs, weight.node)
        )

    def test_a_structural_graph_compiles_into_a_program(self):
        inputs = VariableNode()
        weight = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")
        bias = ts.Variable([0.5, 0.5], name="bias")

        output = ts.relu((inputs @ weight) + bias)

        compiler = Compiler((output,), boundaries=(inputs,))
        instructions = compiler.compile()

        self.assertEqual(
            [instruction.operation.name for instruction in instructions],
            ["dot", "add", "relu"],
        )
        self.assertEqual(
            compiler.output_slots, (compiler.node_slots[output],)
        )
        # The model input and the parameters are the values the program
        # reads; everything else it produces.
        self.assertEqual(
            sorted(compiler.leaf_slots),
            sorted(
                compiler.node_slots[node]
                for node in (inputs, weight.node, bias.node)
            ),
        )
        self.assertFalse(output.is_bound)

    def test_structural_application_neither_executes_nor_compiles(self):
        inputs = VariableNode()
        weight = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="weight")
        bias = ts.Variable([0.5, 0.5], name="bias")
        compilations = []
        executions = []

        def counted_compile(self):
            compilations.append(self)
            raise AssertionError("structural construction must not compile")

        def counted_forward(self, *args):
            executions.append(args)
            raise AssertionError("structural construction must not execute")

        def forbidden(*args, **kwargs):
            raise AssertionError("structural construction must not execute")

        with (
            patch.object(Compiler, "compile", counted_compile),
            patch.object(Computation, "from_nodes", forbidden),
            patch.object(Add, "forward", counted_forward),
            patch.object(Dot, "forward", counted_forward),
            patch.object(ReLU, "forward", counted_forward),
        ):
            output = ts.relu((inputs @ weight) + bias)

        self.assertEqual(compilations, [])
        self.assertEqual(executions, [])
        self.assertFalse(output.is_bound)

    def test_a_python_scalar_operand_is_reported_rather_than_guessed(self):
        node = VariableNode()

        # The dtype a scalar promotes to depends on the value beside it,
        # which a structural expression has not calculated yet.
        with self.assertRaises(TypeError):
            node + 1.0
        with self.assertRaises(TypeError):
            2.0 * node

    def test_a_vertex_is_not_iterable(self):
        node = VariableNode()

        with self.assertRaisesRegex(TypeError, "not iterable"):
            list(node)

    def test_indexing_records_a_slice(self):
        node = VariableNode()

        row = node[0]

        self.assertFalse(row.is_bound)
        self.assertEqual(row.producer.label, "slice")
        self.assertEqual(row.producer.operand_nodes, (node,))

    def test_a_tensor_operand_becomes_a_constant_leaf(self):
        node = VariableNode()

        result = node * ts.Tensor([2.0, 3.0])

        operands = result.producer.operand_nodes
        self.assertIs(operands[0], node)
        self.assertTrue(operands[1].is_bound)
        self.assertEqual(operands[1].variable.data.tolist(), [2.0, 3.0])
        self.assertFalse(operands[1].variable.requires_grad)
        self.assertFalse(result.is_bound)

    def test_structural_application_records_through_the_graph_state(self):
        left = VariableNode()
        right = VariableNode()
        recorded = []
        original = GraphState.record_operation

        def watched(self, operation, inputs):
            recorded.append((operation, tuple(inputs)))
            return original(self, operation, inputs)

        with patch.object(GraphState, "record_operation", watched):
            total = left + right

        (operation, inputs), = recorded
        self.assertIsInstance(operation, Add)
        self.assertEqual(inputs, (left, right))
        self.assertIs(total.producer.operation, operation)


class RuntimeApplicationIsUnchangedTests(unittest.TestCase):
    """Runtime expressions still calculate through a Computation."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_variable_expressions_still_execute(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")

        total = left + right
        product = total * right

        self.assertIsInstance(product, ts.Variable)
        self.assertTrue(product.node.is_bound)
        self.assertEqual(total.data.tolist(), [4.0, 6.0])
        self.assertEqual(product.data.tolist(), [12.0, 24.0])

    def test_public_functions_still_calculate_for_variables(self):
        value = ts.Variable([[-1.0, 2.0], [3.0, -4.0]], name="value")
        weight = ts.Variable([[1.0, 0.0], [0.0, 1.0]], name="weight")

        self.assertEqual(ts.relu(value).data.tolist(), [0.0, 2.0, 3.0, 0.0])
        self.assertEqual(
            ts.matmul(value, weight).data.tolist(), [-1.0, 2.0, 3.0, -4.0]
        )

    def test_backward_still_reaches_the_leaves(self):
        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")

        ts.backward(ts.sum(ts.relu((left + right) * right)))

        self.assertEqual(left.grad.tolist(), [3.0, 4.0])
        self.assertEqual(right.grad.tolist(), [7.0, 10.0])


class TensorOperandBoundaryTests(unittest.TestCase):
    """The graph layer decides what an executing operation may run on."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_the_boundary_rejects_a_vertex(self):
        with self.assertRaises(UnsupportedStructuralExpression):
            as_tensor_operand(VariableNode())

    def test_the_boundary_still_coerces_ordinary_data(self):
        tensor = ts.Tensor([1.0, 2.0])

        self.assertIs(as_tensor_operand(tensor), tensor)
        self.assertEqual(as_tensor_operand([1.0, 2.0]).tolist(), [1.0, 2.0])
        self.assertEqual(as_tensor_operand(3.0).item(), 3.0)
        self.assertEqual(
            as_tensor_operand(3, dtype=ts.float32).dtype, ts.float32
        )

    def test_an_operation_without_a_structural_form_signals(self):
        vertex = VariableNode()
        calls = {
            "sigmoid": lambda: ts.sigmoid(vertex),
            "sum": lambda: ts.sum(vertex),
            "transpose": lambda: ts.transpose(vertex),
            "concat": lambda: ts.concat([vertex, ts.Tensor([1.0])]),
            "where": lambda: ts.where(
                ts.Tensor([True]), vertex, ts.Tensor([1.0])
            ),
            "maximum": lambda: ts.maximum(vertex, ts.Tensor([1.0])),
            "conv1d": lambda: ts.conv1d(vertex, ts.ones((1, 1, 2))),
        }
        for name, call in calls.items():
            with self.subTest(operation=name):
                with self.assertRaises(UnsupportedStructuralExpression):
                    call()

    def test_eager_tensor_behaviour_is_unchanged(self):
        tensor = ts.Tensor([[-1.0, 2.0]])

        self.assertEqual(ts.relu(tensor).tolist(), [0.0, 2.0])
        self.assertEqual(ts.sigmoid(ts.Tensor([0.0])).tolist(), [0.5])
        self.assertEqual(ts.sum([1.0, 2.0, 3.0]).item(), 6.0)
        self.assertEqual(ts.transpose(tensor).shape, (2, 1))
        self.assertEqual(
            ts.concat([[1.0], ts.Tensor([2.0])]).tolist(), [1.0, 2.0]
        )
        with self.assertRaisesRegex(TypeError, "Unsupported data type"):
            ts.Tensor(object())


if __name__ == "__main__":
    unittest.main()
