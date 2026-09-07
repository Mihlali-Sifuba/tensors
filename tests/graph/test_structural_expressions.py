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
from tensors.linalg.norm import Norm
from tensors.linalg.transpose import Transpose
from tensors.linalg.outer import Outer
from tensors.math.abs import Abs
from tensors.math.arccos import ArcCos
from tensors.math.arccosh import ArcCosh
from tensors.math.arcsin import ArcSin
from tensors.math.arcsinh import ArcSinh
from tensors.math.arctan import ArcTan
from tensors.math.arctanh import ArcTanh
from tensors.math.cos import Cos
from tensors.math.cosh import Cosh
from tensors.math.exp import Exp
from tensors.math.concat import Concat
from tensors.math.log import Log
from tensors.math.log_softmax import LogSoftmax
from tensors.math.logsumexp import LogSumExp
from tensors.math.max import Max
from tensors.math.mean import Mean
from tensors.math.min import Min
from tensors.math.prod import Prod
from tensors.math.reshape import Reshape
from tensors.math.softmax import Softmax
from tensors.math.stack import Stack
from tensors.math.std import Std
from tensors.math.sum import Sum
from tensors.math.variance import Variance
from tensors.math.relu import ReLU
from tensors.math.sign import Sign
from tensors.math.sin import Sin
from tensors.math.sinh import Sinh
from tensors.math.softplus import Softplus
from tensors.math.sqrt import Sqrt
from tensors.math.tan import Tan
from tensors.math.tanh import Tanh
from tensors.math.sigmoid import Sigmoid
from tensors.ops import Add, Div, Mul, Pow, Sub


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

    def test_a_tensor_on_the_left_records_through_the_reflected_operator(self):
        """Tensor arithmetic defers a vertex to the operator protocol.

        A Tensor cannot evaluate an expression against a value that does not
        exist yet, so it declines the operation and the vertex records it
        through its reflected operator instead.
        """
        operators = (
            ("+", lambda left, right: left + right, Add),
            ("-", lambda left, right: left - right, Sub),
            ("*", lambda left, right: left * right, Mul),
            ("/", lambda left, right: left / right, Div),
            ("**", lambda left, right: left ** right, Pow),
        )
        for symbol, apply, operation in operators:
            with self.subTest(operator=symbol):
                reset_graph_state()
                node = VariableNode()
                calls = []
                original = operation.forward

                def counted(self, *args, _original=original, _calls=calls):
                    _calls.append(args)
                    return _original(self, *args)

                with patch.object(operation, "forward", counted):
                    result = apply(ts.Tensor([2.0, 4.0]), node)

                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                self.assertIsInstance(result.producer.operation, operation)
                # The vertex keeps its place as the right-hand operand.
                operands = result.producer.operand_nodes
                self.assertIs(operands[1], node)
                # The Tensor took part as a non-gradient leaf.
                self.assertTrue(operands[0].is_bound)
                self.assertFalse(operands[0].variable.requires_grad)
                # Nothing was calculated.
                self.assertEqual(calls, [])

    def test_outer_records_structurally_in_either_position(self):
        """An outer product records whichever side names a graph value."""
        operands = (
            ("vertex, Tensor", lambda node: (node, ts.Tensor([3.0, 4.0]))),
            ("Tensor, vertex", lambda node: (ts.Tensor([3.0, 4.0]), node)),
            ("vertex, Variable",
             lambda node: (node, ts.Variable([3.0, 4.0], name="right"))),
            ("Variable, vertex",
             lambda node: (ts.Variable([3.0, 4.0], name="left"), node)),
            ("vertex, vertex", lambda node: (node, VariableNode())),
            ("vertex, data", lambda node: (node, [3.0, 4.0])),
        )
        for label, build in operands:
            with self.subTest(operands=label):
                reset_graph_state()
                node = VariableNode()
                calls = []
                original = Outer.forward

                def counted(self, *args, _original=original, _calls=calls):
                    _calls.append(args)
                    return _original(self, *args)

                left, right = build(node)
                with patch.object(Outer, "forward", counted):
                    result = ts.outer(left, right)

                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                self.assertIsInstance(result.producer.operation, Outer)
                self.assertEqual(len(result.producer.operand_nodes), 2)
                self.assertIn(node, result.producer.operand_nodes)
                # Nothing was calculated.
                self.assertEqual(calls, [])

    def test_norm_records_structurally_with_its_configuration(self):
        """A recorded norm keeps the reduction the call was written with."""
        configurations = (
            ({}, None, False),
            ({"axis": 1}, 1, False),
            ({"axis": 0, "keepdims": True}, 0, True),
        )
        for keywords, axis, keepdims in configurations:
            with self.subTest(**keywords):
                reset_graph_state()
                node = VariableNode()
                calls = []
                original = Norm.forward

                def counted(self, *args, _original=original, _calls=calls):
                    _calls.append(args)
                    return _original(self, *args)

                with patch.object(Norm, "forward", counted):
                    result = ts.norm(node, **keywords)

                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                operation = result.producer.operation
                self.assertIsInstance(operation, Norm)
                # The configuration is the operation's, not the caller's.
                self.assertEqual(operation.axis, axis)
                self.assertEqual(operation.keepdims, keepdims)
                self.assertEqual(result.producer.operand_nodes, (node,))
                # The vertex names no value, and none was read.
                self.assertEqual(calls, [])

    def test_pow_records_structurally_in_both_operand_orders(self):
        reset_graph_state()
        base_vertex = ts.pow(VariableNode(), ts.Tensor([2.0, 2.0]))
        reset_graph_state()
        exponent_vertex = ts.pow(ts.Tensor([2.0, 2.0]), VariableNode())

        for label, result in (
            ("base", base_vertex), ("exponent", exponent_vertex)
        ):
            with self.subTest(vertex=label):
                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                self.assertIsInstance(result.producer.operation, Pow)

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

    def test_sigmoid_records_structurally(self):
        node = VariableNode()
        calls = []
        original = Sigmoid.forward

        def counted(self, *args):
            calls.append(args)
            return original(self, *args)

        with patch.object(Sigmoid, "forward", counted):
            activated = ts.sigmoid(node)

        self.assertIsInstance(activated, VariableNode)
        self.assertFalse(activated.is_bound)
        self.assertIsInstance(activated.producer.operation, Sigmoid)
        self.assertEqual(activated.producer.operand_nodes, (node,))
        # The vertex names a value that does not exist yet, so the numerical
        # forward never ran.
        self.assertEqual(calls, [])

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


class UnaryFamilyStructuralTests(unittest.TestCase):
    """Every elementwise unary function applies across all three kinds."""

    #: Each public unary function, the operation it records, and an input
    #: inside its domain.
    FAMILY = (
        ("abs", Abs, [-0.5, 0.5]),
        ("sign", Sign, [-0.5, 0.5]),
        ("sin", Sin, [0.5, 0.25]),
        ("cos", Cos, [0.5, 0.25]),
        ("tan", Tan, [0.5, 0.25]),
        ("sinh", Sinh, [0.5, 0.25]),
        ("cosh", Cosh, [0.5, 0.25]),
        ("tanh", Tanh, [0.5, 0.25]),
        ("exp", Exp, [0.5, 0.25]),
        ("log", Log, [1.0, 2.0]),
        ("sqrt", Sqrt, [1.0, 4.0]),
        ("softplus", Softplus, [0.5, 0.25]),
        ("arcsin", ArcSin, [0.5, 0.25]),
        ("arccos", ArcCos, [0.5, 0.25]),
        ("arctan", ArcTan, [0.5, 0.25]),
        ("arcsinh", ArcSinh, [0.5, 0.25]),
        ("arccosh", ArcCosh, [1.5, 2.5]),
        ("arctanh", ArcTanh, [0.5, 0.25]),
    )

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_the_family_is_the_whole_exported_unary_surface(self):
        # A function added to the family without a case here would go
        # uncovered, so the table states its own completeness.
        self.assertEqual(len(self.FAMILY), 18)
        for name, operation, _ in self.FAMILY:
            with self.subTest(function=name):
                self.assertTrue(hasattr(ts, name))
                self.assertEqual(operation().name, name)

    def test_a_vertex_records_the_operation(self):
        for name, operation, _ in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                node = VariableNode()
                calls = []
                original = operation.forward

                def counted(self, *args, _original=original, _calls=calls):
                    _calls.append(args)
                    return _original(self, *args)

                with patch.object(operation, "forward", counted):
                    result = getattr(ts, name)(node)

                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                self.assertIsInstance(result.producer.operation, operation)
                self.assertEqual(result.producer.operand_nodes, (node,))
                # Nothing was calculated, and the vertex still names a
                # value that does not exist.
                self.assertEqual(calls, [])
                self.assertFalse(node.is_bound)

    def test_a_tensor_still_calculates_eagerly(self):
        for name, _, sample in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                result = getattr(ts, name)(ts.Tensor(sample))
                self.assertIsInstance(result, ts.Tensor)
                self.assertEqual(result.shape, (2,))

    def test_a_variable_still_calculates_and_differentiates(self):
        for name, _, sample in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                function = getattr(ts, name)
                expected = function(ts.Tensor(sample))
                variable = ts.Variable(sample, name="value")
                result = function(variable)

                self.assertIsInstance(result, ts.Variable)
                self.assertTrue(result.node.is_bound)
                # The graph path calculates exactly what the Tensor path does.
                self.assertEqual(result.data.tolist(), expected.tolist())

                ts.backward(ts.sum(result))
                self.assertIsNotNone(variable.grad)
                self.assertEqual(variable.grad.shape, variable.shape)


class ReductionFamilyStructuralTests(unittest.TestCase):
    """A reduction or normalization records the call it was configured by."""

    #: Each public reduction and the operation it records.
    REDUCTIONS = (
        ("sum", Sum), ("mean", Mean), ("prod", Prod), ("max", Max),
        ("min", Min), ("std", Std), ("variance", Variance),
        ("logsumexp", LogSumExp),
    )
    #: Each public normalization and the operation it records.
    NORMALIZATIONS = (("softmax", Softmax), ("log_softmax", LogSoftmax))

    #: Reduction keywords paired with the configuration they must record.
    CONFIGURATIONS = (
        ({}, None, False),
        ({"axis": 0}, 0, False),
        ({"axis": 1}, 1, False),
        ({"axis": 1, "keepdims": True}, 1, True),
        ({"axis": (0, 1)}, (0, 1), False),
        ({"axis": (0, 1), "keepdims": True}, (0, 1), True),
    )

    VALUES = [[1.0, 2.0], [3.0, 4.0]]

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_the_tables_cover_the_migrated_surface(self):
        self.assertEqual(len(self.REDUCTIONS), 8)
        self.assertEqual(len(self.NORMALIZATIONS), 2)
        for name, operation in self.REDUCTIONS + self.NORMALIZATIONS:
            with self.subTest(function=name):
                self.assertTrue(hasattr(ts, name))
                self.assertEqual(operation().name, name)

    def test_a_reduction_records_its_configuration(self):
        for name, operation in self.REDUCTIONS:
            for keywords, axis, keepdims in self.CONFIGURATIONS:
                with self.subTest(function=name, **keywords):
                    reset_graph_state()
                    node = VariableNode()
                    calls = []
                    original = operation.forward

                    def counted(self, *args, _original=original, _calls=calls):
                        _calls.append(args)
                        return _original(self, *args)

                    with patch.object(operation, "forward", counted):
                        result = getattr(ts, name)(node, **keywords)

                    self.assertIsInstance(result, VariableNode)
                    self.assertFalse(result.is_bound)
                    recorded = result.producer.operation
                    self.assertIsInstance(recorded, operation)
                    # The reduction is the operation's own state, recorded
                    # exactly as the call configured it.
                    self.assertEqual(recorded.axis, axis)
                    self.assertEqual(recorded.keepdims, keepdims)
                    self.assertEqual(result.producer.operand_nodes, (node,))
                    # Nothing ran, and the vertex still names no value.
                    self.assertEqual(calls, [])
                    self.assertFalse(node.is_bound)

    def test_a_normalization_records_its_axis(self):
        for name, operation in self.NORMALIZATIONS:
            for axis in (-1, 0, 1):
                with self.subTest(function=name, axis=axis):
                    reset_graph_state()
                    node = VariableNode()
                    calls = []
                    original = operation.forward

                    def counted(self, *args, _original=original, _calls=calls):
                        _calls.append(args)
                        return _original(self, *args)

                    with patch.object(operation, "forward", counted):
                        result = getattr(ts, name)(node, axis=axis)

                    self.assertIsInstance(result, VariableNode)
                    self.assertFalse(result.is_bound)
                    recorded = result.producer.operation
                    self.assertIsInstance(recorded, operation)
                    self.assertEqual(recorded.axis, axis)
                    self.assertEqual(result.producer.operand_nodes, (node,))
                    self.assertEqual(calls, [])
                    self.assertFalse(node.is_bound)

    def test_a_recorded_program_replays_what_eager_calculates(self):
        values = ts.Tensor(self.VALUES)
        cases = [
            (name, keywords)
            for name, _ in self.REDUCTIONS
            for keywords, _, _ in self.CONFIGURATIONS
        ] + [
            (name, {"axis": axis})
            for name, _ in self.NORMALIZATIONS
            for axis in (-1, 0, 1)
        ]
        for name, keywords in cases:
            with self.subTest(function=name, **keywords):
                reset_graph_state()
                function = getattr(ts, name)
                expected = function(values, **keywords)

                node = VariableNode()
                output = function(node, **keywords)
                compiler = Compiler((output,), boundaries=(node,))
                compiler.compile()
                computations = Computation._from_compiler(compiler)
                node.materialize(values, requires_grad=False)
                for computation in computations:
                    computation.forward()

                replayed = output.variable.data
                # Replay is the same program, so it is exact rather than close.
                self.assertEqual(replayed.tolist(), expected.tolist())
                self.assertIs(replayed.dtype, expected.dtype)
                self.assertEqual(replayed.shape, expected.shape)

    def test_eager_application_is_unchanged(self):
        values = ts.Tensor(self.VALUES)
        for name, _ in self.REDUCTIONS + self.NORMALIZATIONS:
            with self.subTest(function=name):
                reset_graph_state()
                function = getattr(ts, name)
                tensor_result = function(values)
                self.assertIsInstance(tensor_result, ts.Tensor)

                variable = ts.Variable(self.VALUES, name="value")
                variable_result = function(variable)
                self.assertIsInstance(variable_result, ts.Variable)
                self.assertEqual(
                    variable_result.data.tolist(), tensor_result.tolist()
                )

                ts.backward(ts.sum(variable_result))
                self.assertIsNotNone(variable.grad)
                self.assertEqual(variable.grad.shape, variable.shape)


class ShapeStructuralTests(unittest.TestCase):
    """reshape and transpose record the layout they were configured with."""

    #: A call, the operation it records, and the state that operation holds.
    CASES = (
        ("reshape tuple",
         lambda value: ts.reshape(value, (3, 2)), Reshape, "shape", (3, 2)),
        ("reshape list",
         lambda value: ts.reshape(value, [2, 3]), Reshape, "shape", (2, 3)),
        ("transpose default",
         lambda value: ts.transpose(value), Transpose, "axes", None),
        ("transpose tuple",
         lambda value: ts.transpose(value, (1, 0)), Transpose, "axes", (1, 0)),
        ("transpose list",
         lambda value: ts.transpose(value, [1, 0]), Transpose, "axes", (1, 0)),
    )

    VALUES = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_vertex_records_the_configured_operation(self):
        for label, apply, operation, attribute, expected in self.CASES:
            with self.subTest(case=label):
                reset_graph_state()
                node = VariableNode()
                calls = []
                original = operation.forward

                def counted(self, *args, _original=original, _calls=calls):
                    _calls.append(args)
                    return _original(self, *args)

                with patch.object(operation, "forward", counted):
                    result = apply(node)

                self.assertIsInstance(result, VariableNode)
                self.assertFalse(result.is_bound)
                recorded = result.producer.operation
                self.assertIsInstance(recorded, operation)
                # The layout is the operation's own state, normalized the
                # way the eager call normalizes it.
                self.assertEqual(getattr(recorded, attribute), expected)
                self.assertEqual(result.producer.operand_nodes, (node,))
                self.assertEqual(calls, [])
                self.assertFalse(node.is_bound)

    def test_a_recorded_program_replays_what_eager_calculates(self):
        values = ts.Tensor(self.VALUES)
        for label, apply, _, _, _ in self.CASES:
            with self.subTest(case=label):
                reset_graph_state()
                expected = apply(values)

                node = VariableNode()
                output = apply(node)
                compiler = Compiler((output,), boundaries=(node,))
                compiler.compile()
                computations = Computation._from_compiler(compiler)
                node.materialize(values, requires_grad=False)
                for computation in computations:
                    computation.forward()

                replayed = output.variable.data
                self.assertEqual(replayed.tolist(), expected.tolist())
                self.assertEqual(replayed.shape, expected.shape)
                self.assertIs(replayed.dtype, expected.dtype)

    def test_eager_application_is_unchanged(self):
        values = ts.Tensor(self.VALUES)
        for label, apply, _, _, _ in self.CASES:
            with self.subTest(case=label):
                reset_graph_state()
                tensor_result = apply(values)
                self.assertIsInstance(tensor_result, ts.Tensor)

                variable = ts.Variable(self.VALUES, name="value")
                variable_result = apply(variable)
                self.assertIsInstance(variable_result, ts.Variable)
                self.assertEqual(
                    variable_result.data.tolist(), tensor_result.tolist()
                )

                ts.backward(ts.sum(variable_result))
                self.assertIsNotNone(variable.grad)
                self.assertEqual(variable.grad.shape, variable.shape)


class SequenceStructuralTests(unittest.TestCase):
    """concat and stack are asked about each operand, not the sequence."""

    FAMILY = (("concat", Concat), ("stack", Stack))

    #: Each operand sequence built around one vertex, and the positions the
    #: vertices occupy in it.
    LAYOUTS = (
        ("[vertex, Tensor]",
         lambda node: [node, ts.Tensor([1.0])], (0,)),
        ("[Tensor, vertex]",
         lambda node: [ts.Tensor([1.0]), node], (1,)),
        ("[vertex, Variable]",
         lambda node: [node, ts.Variable([2.0], name="right")], (0,)),
        ("[Variable, vertex]",
         lambda node: [ts.Variable([2.0], name="left"), node], (1,)),
        ("[vertex, vertex]",
         lambda node: [node, VariableNode()], (0, 1)),
        ("[Tensor, vertex, Variable]",
         lambda node: [
             ts.Tensor([1.0]), node, ts.Variable([3.0], name="third")
         ], (1,)),
        ("(vertex, Tensor) tuple",
         lambda node: (node, ts.Tensor([1.0])), (0,)),
        ("[vertex, data]",
         lambda node: [node, [1.0]], (0,)),
    )

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_sequence_holding_a_vertex_records_in_order(self):
        for name, operation in self.FAMILY:
            for label, build, positions in self.LAYOUTS:
                with self.subTest(function=name, operands=label):
                    reset_graph_state()
                    node = VariableNode()
                    sequence = build(node)
                    calls = []
                    original = operation.forward

                    def counted(self, *args, _original=original, _calls=calls):
                        _calls.append(args)
                        return _original(self, *args)

                    with patch.object(operation, "forward", counted):
                        result = getattr(ts, name)(sequence, axis=0)

                    self.assertIsInstance(result, VariableNode)
                    self.assertFalse(result.is_bound)
                    self.assertIsInstance(result.producer.operation, operation)

                    operands = result.producer.operand_nodes
                    # One operand per element, in the order given.
                    self.assertEqual(len(operands), len(sequence))
                    self.assertIs(operands[positions[0]], node)
                    for index, operand in enumerate(operands):
                        if index in positions:
                            # A vertex names a value that does not exist.
                            self.assertFalse(operand.is_bound)
                        else:
                            # A runtime operand beside it is a bound leaf.
                            self.assertTrue(operand.is_bound)
                    self.assertEqual(calls, [])
                    self.assertFalse(node.is_bound)

    def test_a_tensor_operand_becomes_a_non_gradient_leaf(self):
        for name, _ in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                node = VariableNode()
                parameter = ts.Variable([2.0], name="parameter")
                result = getattr(ts, name)(
                    [node, parameter, ts.Tensor([9.0])], axis=0
                )

                first, second, third = result.producer.operand_nodes
                self.assertIs(first, node)
                # A Variable keeps its own gradient flag; a Tensor does not.
                self.assertIs(second.variable, parameter)
                self.assertTrue(second.variable.requires_grad)
                self.assertTrue(third.is_bound)
                self.assertFalse(third.variable.requires_grad)

    def test_a_recorded_sequence_replays_what_eager_calculates(self):
        left = ts.Tensor([[1.0, 2.0]])
        right = ts.Tensor([[3.0, 4.0]])
        for name, _ in self.FAMILY:
            for axis in (0, 1):
                with self.subTest(function=name, axis=axis):
                    reset_graph_state()
                    function = getattr(ts, name)
                    expected = function([left, right], axis=axis)

                    node = VariableNode()
                    output = function([node, right], axis=axis)
                    compiler = Compiler((output,), boundaries=(node,))
                    compiler.compile()
                    computations = Computation._from_compiler(compiler)
                    node.materialize(left, requires_grad=False)
                    for computation in computations:
                        computation.forward()

                    replayed = output.variable.data
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_a_sequence_without_a_vertex_is_unchanged(self):
        for name, _ in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                function = getattr(ts, name)
                left = ts.Variable([[1.0, 2.0]], name="left")
                right = ts.Variable([[3.0, 4.0]], name="right")

                tensors = function(
                    [ts.Tensor([[1.0, 2.0]]), ts.Tensor([[3.0, 4.0]])]
                )
                self.assertIsInstance(tensors, ts.Tensor)

                variables = function([left, right])
                self.assertIsInstance(variables, ts.Variable)
                self.assertEqual(variables.data.tolist(), tensors.tolist())

                ts.backward(ts.sum(function([left, right])))
                self.assertIsNotNone(left.grad)
                self.assertIsNotNone(right.grad)

    def test_an_invalid_sequence_still_fails(self):
        # Structural dispatch must not let a malformed sequence through.
        for name, _ in self.FAMILY:
            with self.subTest(function=name):
                reset_graph_state()
                with self.assertRaises(ValueError):
                    getattr(ts, name)([])


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

    def test_sigmoid_still_applies_to_both_runtime_kinds(self):
        tensor = ts.sigmoid(ts.Tensor([0.0, 0.0]))

        self.assertIsInstance(tensor, ts.Tensor)
        self.assertEqual(tensor.tolist(), [0.5, 0.5])

        value = ts.Variable([0.0, 0.0], name="value")
        result = ts.sigmoid(value)

        self.assertIsInstance(result, ts.Variable)
        self.assertTrue(result.node.is_bound)
        self.assertEqual(result.data.tolist(), [0.5, 0.5])

        ts.backward(ts.sum(result))

        for gradient in value.grad.tolist():
            self.assertAlmostEqual(gradient, 0.25)

    def test_outer_and_norm_still_apply_to_the_runtime_kinds(self):
        self.assertEqual(
            ts.outer([1.0, 2.0], [3.0, 4.0]).tolist(), [3.0, 4.0, 6.0, 8.0]
        )
        self.assertIsInstance(ts.outer([1.0, 2.0], [3.0, 4.0]), ts.Tensor)
        self.assertEqual(ts.norm([3.0, 4.0]).item(), 5.0)
        self.assertIsInstance(ts.norm([3.0, 4.0]), ts.Tensor)

        left = ts.Variable([1.0, 2.0], name="left")
        right = ts.Variable([3.0, 4.0], name="right")
        product = ts.outer(left, right)
        magnitude = ts.norm(ts.Variable([3.0, 4.0], name="vector"))

        self.assertIsInstance(product, ts.Variable)
        self.assertTrue(product.node.is_bound)
        self.assertEqual(product.data.tolist(), [3.0, 4.0, 6.0, 8.0])
        self.assertIsInstance(magnitude, ts.Variable)
        self.assertEqual(magnitude.data.item(), 5.0)

        ts.backward(ts.sum(ts.outer(left, right)))
        self.assertEqual(left.grad.tolist(), [7.0, 7.0])
        self.assertEqual(right.grad.tolist(), [3.0, 3.0])

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
        self.assertEqual(
            ts.stack([[1.0], ts.Tensor([2.0])]).shape, (2, 1)
        )
        with self.assertRaisesRegex(TypeError, "Unsupported data type"):
            ts.Tensor(object())


if __name__ == "__main__":
    unittest.main()
