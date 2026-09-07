import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph.computation.compiler import Compiler
from tensors.graph.expression import UnsupportedStructuralExpression
from tensors.graph.node import VariableNode
from tensors.graph.state import reset_graph_state
from tensors.ops import Add, Mul


class Linear(ts.Graph):
    """The model shape this lifecycle is built for."""

    def __init__(self):
        super().__init__()
        self.w = ts.Variable([[2.0]], name="w")
        self.b = ts.Variable([1.0], name="b")

    def forward(self, x):
        return x @ self.w + self.b


class ModelConstructionTests(unittest.TestCase):
    """A subclass builds and compiles its graph as it is constructed."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_construction_builds_the_structural_graph(self):
        model = Linear()

        structure = model._structure
        self.assertIsNotNone(structure)
        self.assertEqual(len(structure.inputs), 1)
        self.assertIsInstance(structure.inputs[0], VariableNode)
        # The graph exists before any value does.
        self.assertFalse(structure.inputs[0].is_bound)
        self.assertEqual(
            [node.label for node in model.nodes],
            ["var", "var", "dot", "var", "var", "add", "var"],
        )

    def test_parameters_take_part_as_their_own_vertices(self):
        model = Linear()

        product = model._structure.computations[0]._variable_nodes
        self.assertIn(model.w.node, product)
        self.assertIn(model.b.node, product)
        # The build used the parameters that __init__ created, not copies.
        self.assertIs(model.w.node.variable, model.w)
        self.assertIs(model.b.node.variable, model.b)

    def test_construction_executes_no_kernels(self):
        executed = []
        originals = {Add: Add.forward, Mul: Mul.forward}

        def counted(self, *args):
            executed.append(type(self).__name__)
            return originals[type(self)](self, *args)

        with patch.object(Add, "forward", counted):
            model = Linear()

        self.assertEqual(executed, [])
        self.assertFalse(model._structure.inputs[0].is_bound)

    def test_the_model_is_compiled_once_during_construction(self):
        compilations = []
        original = Compiler.compile

        def counted(self):
            compilations.append(self)
            return original(self)

        with patch.object(Compiler, "compile", counted):
            model = Linear()

        self.assertEqual(len(compilations), 1)
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in compilations[0].instructions
            ],
            ["dot", "add"],
        )
        self.assertIs(
            model._structure.computations[0]._instructions,
            compilations[0].instructions,
        )


class ModelExecutionTests(unittest.TestCase):
    """Calling a built model binds its inputs and replays its program."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_call_executes_the_prebuilt_computation(self):
        model = Linear()
        computation = model._structure.computations[0]

        result = model(ts.Tensor([[3.0]]))

        self.assertIs(model.computation, computation)
        self.assertIs(computation.output, result)
        self.assertEqual(result.data.tolist(), [7.0])
        # The call bound the input vertex the graph was built with.
        self.assertTrue(model._structure.inputs[0].is_bound)
        self.assertEqual(
            model._structure.inputs[0].variable.data.tolist(), [3.0]
        )

    def test_a_call_does_not_re_execute_the_python_forward(self):
        calls = []

        class Counted(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0]], name="w")

            def forward(self, x):
                calls.append(x)
                return x @ self.w

        model = Counted()
        self.assertEqual(len(calls), 1)

        model(ts.Tensor([[3.0]]))
        model(ts.Tensor([[4.0]]))

        # Only the structural build ran forward; the calls replayed it.
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0], VariableNode)

    def test_repeated_calls_update_the_input_and_replay(self):
        model = Linear()
        structure = model._structure

        first = model(ts.Tensor([[3.0]]))
        second = model(ts.Tensor([[4.0]]))

        self.assertIs(first, second)
        self.assertEqual(second.data.tolist(), [9.0])
        self.assertIs(model._structure, structure)
        self.assertIs(
            model._structure.computations[0], structure.computations[0]
        )

    def test_structural_identity_is_stable_across_calls(self):
        model = Linear()
        nodes = list(model.nodes)
        output_node = model._structure.computations[0]._variable_nodes[-1]

        model(ts.Tensor([[3.0]]))
        model(ts.Tensor([[5.0]]))

        self.assertEqual(list(model.nodes), nodes)
        self.assertIs(
            model._structure.computations[0]._variable_nodes[-1], output_node
        )
        self.assertIs(output_node.variable, model.computation.output)

    def test_outputs_are_numerically_correct(self):
        model = Linear()

        self.assertEqual(model(ts.Tensor([[3.0]])).data.tolist(), [7.0])
        self.assertEqual(model(ts.Tensor([[0.5]])).data.tolist(), [2.0])
        self.assertEqual(model(ts.Tensor([[-2.0]])).data.tolist(), [-3.0])

    def test_gradients_still_reach_the_parameters(self):
        model = Linear()

        prediction = model(ts.Tensor([[3.0]]))
        ts.backward(ts.sum(prediction))

        self.assertEqual(model.w.grad.tolist(), [3.0])
        self.assertEqual(model.b.grad.tolist(), [1.0])
        self.assertEqual(model.parameters(), [model.w, model.b])

    def test_a_nested_model_contributes_its_operations(self):
        class Inner(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[3.0]], name="inner_w")

            def forward(self, x):
                return x @ self.w

        class Outer(ts.Graph):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.b = ts.Variable([1.0], name="outer_b")

            def forward(self, x):
                return self.inner(x) + self.b

        model = Outer()

        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "add"],
        )
        result = model(ts.Tensor([[2.0]]))
        self.assertEqual(result.data.tolist(), [7.0])
        ts.backward(ts.sum(result))
        self.assertEqual(model.inner.w.grad.tolist(), [2.0])
        self.assertEqual(model.b.grad.tolist(), [1.0])
        self.assertEqual(model.parameters(), [model.inner.w, model.b])

    def test_a_functional_graph_keeps_tracing(self):
        weight = ts.Variable([2.0], name="weight")

        @ts.Graph
        def model(x):
            return x * weight + 1.0

        self.assertIsNone(model._structure)
        first = model(ts.Tensor([3.0]))
        second = model(ts.Tensor([4.0]))

        # The functional form still records a fresh computation per call.
        self.assertIsNot(first, second)
        self.assertEqual(first.data.tolist(), [7.0])
        self.assertEqual(second.data.tolist(), [9.0])

    def test_a_model_needing_runtime_values_keeps_tracing(self):
        class Scaled(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([2.0], name="w")

            def forward(self, x):
                # A Python scalar cannot be recorded before the value beside
                # it exists, so this model keeps the tracing lifecycle.
                return x * self.w + 1.0

        model = Scaled()

        self.assertIsNone(model._structure)
        self.assertEqual(model(ts.Tensor([3.0])).data.tolist(), [7.0])
        self.assertEqual(model(ts.Tensor([4.0])).data.tolist(), [9.0])


class StructuralBuildFallbackTests(unittest.TestCase):
    """Only an expression the graph cannot record yet falls back to tracing."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_scalar_operand_keeps_the_tracing_lifecycle(self):
        class Scaled(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([2.0], name="w")

            def forward(self, x):
                return x * self.w + 1.0

        model = Scaled()

        self.assertIsNone(model._structure)
        self.assertEqual(model(ts.Tensor([3.0])).data.tolist(), [7.0])
        # The build stopped on the signal that states the limit.
        with self.assertRaises(UnsupportedStructuralExpression):
            VariableNode() + 1.0

    def test_a_function_without_a_structural_form_keeps_tracing(self):
        class Transposed(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0]], name="w")

            def forward(self, x):
                return ts.transpose(x @ self.w)

        model = Transposed()

        self.assertIsNone(model._structure)
        self.assertEqual(model(ts.Tensor([[1.0]])).data.item(), 2.0)
        with self.assertRaises(UnsupportedStructuralExpression):
            ts.transpose(VariableNode())

    def test_the_signal_is_narrower_than_a_type_error(self):
        # Existing callers still see a TypeError, but the build only treats
        # this one as "cannot be recorded yet".
        self.assertTrue(
            issubclass(UnsupportedStructuralExpression, TypeError)
        )
        with self.assertRaises(TypeError):
            VariableNode() * 2.0

    def test_a_misspelled_model_attribute_is_not_swallowed(self):
        class Broken(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0]], name="w")

            def forward(self, x):
                return x @ self.ww

        with self.assertRaisesRegex(AttributeError, "ww"):
            Broken()

    def test_a_failing_forward_propagates_during_construction(self):
        class Failing(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0]], name="w")

            def forward(self, x):
                raise ValueError("deliberate model failure")

        with self.assertRaisesRegex(ValueError, "deliberate model failure"):
            Failing()

    def test_a_compiler_failure_does_not_become_a_tracing_fallback(self):
        def broken(self):
            raise RuntimeError("deliberate compiler failure")

        with patch.object(Compiler, "compile", broken):
            with self.assertRaisesRegex(RuntimeError, "deliberate compiler"):
                Linear()

    def test_a_malformed_output_is_reported_during_construction(self):
        class NotAGraphValue(ts.Graph):
            def forward(self, x):
                return 5

        with self.assertRaisesRegex(TypeError, "must return"):
            NotAGraphValue()

    def test_a_supported_model_still_builds_during_construction(self):
        model = Linear()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "add"],
        )
        self.assertEqual(model(ts.Tensor([[3.0]])).data.tolist(), [7.0])


class SigmoidModelConstructionTests(unittest.TestCase):
    """A sigmoid activation no longer abandons the structural build."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_a_sigmoid_model_builds_during_construction(self):
        class Activated(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[0.0]], name="w")

            def forward(self, x):
                return ts.sigmoid(x @ self.w)

        model = Activated()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "sigmoid"],
        )
        self.assertEqual(model(ts.Tensor([[1.0]])).data.tolist(), [0.5])

    def test_the_xor_model_shape_builds_during_construction(self):
        class MLP(ts.Graph):
            def __init__(self):
                super().__init__()
                self.hidden_weight = ts.Variable(
                    [[0.5, -0.5], [-0.5, 0.5]], name="hidden_weight"
                )
                self.hidden_bias = ts.Variable([0.0, 0.0], name="hidden_bias")
                self.output_weight = ts.Variable(
                    [[1.0], [1.0]], name="output_weight"
                )
                self.output_bias = ts.Variable([0.0], name="output_bias")

            def forward(self, inputs):
                hidden = ts.relu(
                    inputs @ self.hidden_weight + self.hidden_bias
                )
                logits = hidden @ self.output_weight + self.output_bias
                return ts.sigmoid(logits)

        model = MLP()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "add", "relu", "dot", "add", "sigmoid"],
        )

        predictions = model(ts.Tensor([[-1.0, 1.0], [1.0, 1.0]]))
        self.assertEqual(predictions.shape, (2, 1))

        # The built program is still differentiable end to end.
        ts.backward(ts.sum(predictions))
        for parameter in model.parameters():
            with self.subTest(parameter=parameter.name):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.shape, parameter.shape)


class LinalgModelConstructionTests(unittest.TestCase):
    """An outer product or a norm no longer breaks the structural build."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class OuterModel(ts.Graph):
        def __init__(self):
            super().__init__()
            self.w = ts.Variable([1.0, 2.0, 3.0], name="w")

        def forward(self, x):
            return ts.outer(x, self.w)

    class NormModel(ts.Graph):
        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[2.0], [3.0]], name="w")

        def forward(self, x):
            return ts.norm(x @ self.w)

    def test_an_outer_model_builds_during_construction(self):
        # Constructing this model used to raise a plain TypeError, which the
        # build cannot treat as a fallback signal, so the model could not be
        # created at all.
        model = self.OuterModel()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["outer"],
        )

    def test_a_norm_model_builds_during_construction(self):
        model = self.NormModel()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "norm"],
        )

    def test_the_built_programs_replay_and_differentiate(self):
        outer_model = self.OuterModel()

        self.assertEqual(
            outer_model(ts.Tensor([1.0, 2.0])).data.tolist(),
            [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        )
        # A second call replays the same program with the new input.
        self.assertEqual(
            outer_model(ts.Tensor([10.0, 20.0])).data.tolist(),
            [10.0, 20.0, 30.0, 20.0, 40.0, 60.0],
        )

        norm_model = self.NormModel()
        self.assertEqual(
            norm_model(ts.Tensor([[3.0, 4.0]])).data.tolist(), [18.0]
        )

        for model, inputs in (
            (self.OuterModel(), ts.Tensor([1.0, 2.0])),
            (self.NormModel(), ts.Tensor([[3.0, 4.0]])),
        ):
            with self.subTest(model=type(model).__name__):
                ts.backward(ts.sum(model(inputs)))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(
                        parameter.grad.shape, parameter.shape
                    )

    def test_the_fallback_signal_is_still_the_only_one_caught(self):
        # A function with no structural form keeps the tracing lifecycle.
        class Transposed(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0], [3.0]], name="w")

            def forward(self, x):
                return ts.transpose(x @ self.w)

        traced = Transposed()
        self.assertIsNone(traced._structure)
        self.assertEqual(traced(ts.Tensor([[1.0, 1.0]])).data.item(), 5.0)

        # Anything else is a real error and construction reports it, so the
        # build must not have widened to catch TypeError generally.
        class Failing(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[1.0]], name="w")

            def forward(self, x):
                raise TypeError("a real bug in forward")

        with self.assertRaisesRegex(TypeError, "a real bug in forward"):
            Failing()


class UnaryModelConstructionTests(unittest.TestCase):
    """A model built from elementwise unary functions compiles at once."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Activated(ts.Graph):
        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0], [1.0]], name="w")

        def forward(self, x):
            hidden = ts.tanh(x @ self.w)
            return ts.sqrt(ts.abs(ts.exp(hidden)) + ts.softplus(hidden))

    def test_a_unary_model_builds_during_construction(self):
        model = self.Activated()

        self.assertIsNotNone(model._structure)
        program = model._structure.computations[0]
        self.assertEqual(
            [
                instruction.operation.name
                for instruction in program._instructions
            ],
            ["dot", "tanh", "exp", "abs", "softplus", "add", "sqrt"],
        )

    def test_the_built_program_replays_and_differentiates(self):
        model = self.Activated()

        first = model(ts.Tensor([[0.5, 0.25]])).data.tolist()
        second = model(ts.Tensor([[1.0, 1.0]])).data.tolist()
        self.assertNotEqual(first, second)
        # Replay is numerically the eager result of the same expression.
        hidden = ts.tanh(ts.Tensor([[1.0, 1.0]]) @ model.w.data)
        expected = ts.sqrt(ts.abs(ts.exp(hidden)) + ts.softplus(hidden))
        self.assertEqual(second, expected.tolist())

        ts.backward(ts.sum(model(ts.Tensor([[0.5, 0.25]]))))
        for parameter in model.parameters():
            with self.subTest(parameter=parameter.name):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(parameter.grad.shape, parameter.shape)

    def test_each_unary_function_builds_a_model_of_its_own(self):
        names = (
            "abs", "sign", "sin", "cos", "tan", "sinh", "cosh", "tanh",
            "exp", "log", "sqrt", "softplus", "arcsin", "arccos", "arctan",
            "arcsinh", "arccosh", "arctanh",
        )
        for name in names:
            with self.subTest(function=name):
                reset_graph_state()
                function = getattr(ts, name)

                # The function is captured by closure: an extra parameter
                # with a default would read as a configuration argument and
                # the model would keep tracing instead of building.
                class Single(ts.Graph):
                    def __init__(self):
                        super().__init__()
                        self.w = ts.Variable([[1.0], [1.0]], name="w")

                    def forward(self, x):
                        return function(x @ self.w)

                model = Single()

                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                self.assertEqual(
                    [
                        instruction.operation.name
                        for instruction in program._instructions
                    ],
                    ["dot", name],
                )

    def test_a_later_migration_group_still_keeps_tracing(self):
        # These families are not migrated yet, so a model using one must
        # still fall back rather than build.
        for name, forward in (
            ("transpose", lambda self, x: ts.transpose(x @ self.w)),
            ("maximum", lambda self, x: ts.maximum(x @ self.w, ts.Tensor([1.0]))),
        ):
            with self.subTest(function=name):
                reset_graph_state()
                model = type(
                    "Later",
                    (ts.Graph,),
                    {
                        "__init__": lambda self: (
                            ts.Graph.__init__(self),
                            setattr(
                                self,
                                "w",
                                ts.Variable([[2.0], [3.0]], name="w"),
                            ),
                        )[0],
                        "forward": forward,
                    },
                )()

                self.assertIsNone(model._structure)
                # The model is still usable through eager tracing.
                self.assertIsInstance(
                    model(ts.Tensor([[1.0, 1.0]])), ts.Variable
                )


class ReductionModelConstructionTests(unittest.TestCase):
    """A model reducing or normalizing its values compiles at construction."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Classifier(ts.Graph):
        """dot -> log_softmax, the shape a classifier head is written in."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w")

        def forward(self, x):
            return ts.log_softmax(x @ self.w, axis=-1)

    class Pooled(ts.Graph):
        """dot -> mean over an axis -> softmax over what remains."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w")

        def forward(self, x):
            return ts.softmax(ts.mean(x @ self.w, axis=0), axis=-1)

    class Scored(ts.Graph):
        """A configured reduction reaching a scalar."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w")

        def forward(self, x):
            return ts.sum(ts.std(x @ self.w, axis=1, keepdims=True))

    INPUTS = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])

    def test_the_models_build_during_construction(self):
        expected = {
            "Classifier": ["dot", "log_softmax"],
            "Pooled": ["dot", "mean", "softmax"],
            "Scored": ["dot", "std", "sum"],
        }
        for model in (self.Classifier(), self.Pooled(), self.Scored()):
            with self.subTest(model=type(model).__name__):
                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                self.assertEqual(
                    [
                        instruction.operation.name
                        for instruction in program._instructions
                    ],
                    expected[type(model).__name__],
                )

    def test_the_built_programs_replay_what_eager_calculates(self):
        for factory, forward in (
            (self.Classifier, lambda w, x: ts.log_softmax(x @ w, axis=-1)),
            (self.Pooled, lambda w, x: ts.softmax(ts.mean(x @ w, axis=0), axis=-1)),
            (self.Scored, lambda w, x: ts.sum(ts.std(x @ w, axis=1, keepdims=True))),
        ):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                for inputs in (
                    self.INPUTS,
                    ts.Tensor([[-1.0, 0.5], [2.0, -3.0]]),
                ):
                    replayed = model(inputs).data
                    expected = forward(model.w.data, inputs)
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_gradients_reach_the_parameters_through_the_program(self):
        for factory in (self.Classifier, self.Pooled, self.Scored):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                ts.backward(ts.sum(model(self.INPUTS)))
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.shape, parameter.shape)

    def test_every_migrated_reduction_builds_a_model_of_its_own(self):
        names = (
            "sum", "mean", "prod", "max", "min", "std", "variance",
            "logsumexp", "softmax", "log_softmax",
        )
        for name in names:
            with self.subTest(function=name):
                reset_graph_state()
                function = getattr(ts, name)

                # Captured by closure: a defaulted parameter would read as a
                # configuration argument and the model would keep tracing.
                class Single(ts.Graph):
                    def __init__(self):
                        super().__init__()
                        self.w = ts.Variable(
                            [[1.0, 0.5], [0.5, 1.0]], name="w"
                        )

                    def forward(self, x):
                        return function(x @ self.w)

                model = Single()

                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                self.assertEqual(
                    [
                        instruction.operation.name
                        for instruction in program._instructions
                    ],
                    ["dot", name],
                )


if __name__ == "__main__":
    unittest.main()
