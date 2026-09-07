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
        class Scored(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w")

            def forward(self, x):
                return ts.cross_entropy(
                    x @ self.w, ts.Tensor([0], dtype=ts.int64)
                )

        model = Scored()

        self.assertIsNone(model._structure)
        self.assertIsInstance(
            model(ts.Tensor([[1.0, 2.0]])), ts.Variable
        )
        with self.assertRaises(UnsupportedStructuralExpression):
            ts.cross_entropy(
                VariableNode(), ts.Tensor([0], dtype=ts.int64)
            )

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
        class Scored(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[2.0, 1.0], [3.0, 1.0]], name="w")

            def forward(self, x):
                return ts.cross_entropy(
                    x @ self.w, ts.Tensor([0], dtype=ts.int64)
                )

        traced = Scored()
        self.assertIsNone(traced._structure)
        self.assertIsInstance(
            traced(ts.Tensor([[1.0, 1.0]])), ts.Variable
        )

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
        # argmax and argmin are unsupported too, but they return a
        # Tensor rather than a Variable, so a model cannot end in one.
        # cross_entropy blocks on either target form.
        for name, forward in (
            ("class indices", lambda self, x: ts.cross_entropy(
                x @ self.w, ts.Tensor([0], dtype=ts.int64)
            )),
            ("dense targets", lambda self, x: ts.cross_entropy(
                x @ self.w, ts.Tensor([[1.0]])
            )),
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



class ShapeModelConstructionTests(unittest.TestCase):
    """Shape and sequence operations build a model's program too."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Reshaped(ts.Graph):
        """dot -> reshape -> transpose."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, 0.5, 2.0], [0.5, 1.0, 1.5]], name="w")

        def forward(self, x):
            return ts.transpose(ts.reshape(x @ self.w, (3, 2)))

    class Branched(ts.Graph):
        """Two parameter branches joined by a concatenation."""

        def __init__(self):
            super().__init__()
            self.left = ts.Variable([[1.0], [2.0]], name="left")
            self.right = ts.Variable([[3.0], [4.0]], name="right")

        def forward(self, x):
            return ts.concat([x @ self.left, x @ self.right], axis=1)

    class Stacked(ts.Graph):
        """Two parameter branches joined on a new axis."""

        def __init__(self):
            super().__init__()
            self.left = ts.Variable([[1.0], [2.0]], name="left")
            self.right = ts.Variable([[3.0], [4.0]], name="right")

        def forward(self, x):
            return ts.stack([x @ self.left, x @ self.right], axis=0)

    INPUTS = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])

    def test_the_models_build_during_construction(self):
        expected = {
            "Reshaped": ["dot", "reshape", "transpose"],
            "Branched": ["dot", "dot", "concat"],
            "Stacked": ["dot", "dot", "stack"],
        }
        for model in (self.Reshaped(), self.Branched(), self.Stacked()):
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
            (
                self.Reshaped,
                lambda m, x: ts.transpose(ts.reshape(x @ m.w.data, (3, 2))),
            ),
            (
                self.Branched,
                lambda m, x: ts.concat(
                    [x @ m.left.data, x @ m.right.data], axis=1
                ),
            ),
            (
                self.Stacked,
                lambda m, x: ts.stack(
                    [x @ m.left.data, x @ m.right.data], axis=0
                ),
            ),
        ):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                for inputs in (
                    self.INPUTS,
                    ts.Tensor([[-1.0, 0.5], [2.0, -3.0]]),
                    ts.Tensor([[10.0, -20.0], [0.5, 1.5]]),
                ):
                    replayed = model(inputs).data
                    expected = forward(model, inputs)
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_gradients_reach_every_branch(self):
        for factory in (self.Reshaped, self.Branched, self.Stacked):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                ts.backward(ts.sum(model(self.INPUTS)))
                parameters = model.parameters()
                self.assertTrue(parameters)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.shape, parameter.shape)

    def test_replay_reuses_the_output_variable(self):
        # The identity semantics a built model already has are unchanged by
        # a sequence operation producing the output.
        model = self.Branched()

        first = model(self.INPUTS)
        second = model(ts.Tensor([[3.0, 4.0], [5.0, 6.0]]))

        self.assertIs(first, second)
        self.assertEqual(
            first.data.tolist(),
            ts.concat(
                [
                    ts.Tensor([[3.0, 4.0], [5.0, 6.0]]) @ model.left.data,
                    ts.Tensor([[3.0, 4.0], [5.0, 6.0]]) @ model.right.data,
                ],
                axis=1,
            ).tolist(),
        )

    def test_each_migrated_function_builds_a_model_of_its_own(self):
        cases = (
            ("reshape", lambda self, x: ts.reshape(x @ self.w, (2, 1))),
            ("transpose", lambda self, x: ts.transpose(x @ self.w)),
            ("concat", lambda self, x: ts.concat([x @ self.w, x @ self.w])),
            ("stack", lambda self, x: ts.stack([x @ self.w, x @ self.w])),
        )
        for name, forward in cases:
            with self.subTest(function=name):
                reset_graph_state()
                model = type(
                    "Single",
                    (ts.Graph,),
                    {
                        "__init__": lambda self: (
                            ts.Graph.__init__(self),
                            setattr(
                                self,
                                "w",
                                ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w"),
                            ),
                        )[0],
                        "forward": forward,
                    },
                )()

                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                names = [
                    instruction.operation.name
                    for instruction in program._instructions
                ]
                self.assertEqual(names[-1], name)
                self.assertIsInstance(
                    model(ts.Tensor([[1.0, 2.0]])), ts.Variable
                )



class SelectionModelConstructionTests(unittest.TestCase):
    """Extrema, clipping and selection build a model's program."""

    INPUTS = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
    MASK = ts.Tensor([[1, 0], [0, 1]], dtype=ts.uint8)

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Rectified(ts.Graph):
        """dot -> maximum against a floor parameter."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, -0.5], [-0.5, 1.0]], name="w")
            self.floor = ts.Variable([[0.0, 0.0], [0.0, 0.0]], name="floor")

        def forward(self, x):
            return ts.maximum(x @ self.w, self.floor)

    class Bounded(ts.Graph):
        """dot -> minimum against a constant, then clipped."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0, -0.5], [-0.5, 1.0]], name="w")

        def forward(self, x):
            capped = ts.minimum(x @ self.w, ts.Tensor([[2.0, 2.0]]))
            return ts.clip(capped, -1.0, 1.0)

    class Selected(ts.Graph):
        """A condition arrives as the model's own input vertex."""

        def __init__(self):
            super().__init__()
            self.chosen = ts.Variable([[1.0, 2.0], [3.0, 4.0]], name="chosen")
            self.fallback = ts.Variable(
                [[9.0, 9.0], [9.0, 9.0]], name="fallback"
            )

        def forward(self, condition):
            return ts.where(condition, self.chosen, self.fallback)

    def test_the_models_build_during_construction(self):
        expected = {
            "Rectified": ["dot", "maximum"],
            "Bounded": ["dot", "minimum", "clip"],
            "Selected": ["where"],
        }
        for model in (self.Rectified(), self.Bounded(), self.Selected()):
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
        for factory, forward, inputs in (
            (
                self.Rectified,
                lambda m, x: ts.maximum(x @ m.w.data, m.floor.data),
                (self.INPUTS, ts.Tensor([[-5.0, -6.0], [7.0, -8.0]])),
            ),
            (
                self.Bounded,
                lambda m, x: ts.clip(
                    ts.minimum(x @ m.w.data, ts.Tensor([[2.0, 2.0]])),
                    -1.0,
                    1.0,
                ),
                (self.INPUTS, ts.Tensor([[0.25, -0.5], [1.0, 2.0]])),
            ),
            (
                self.Selected,
                lambda m, c: ts.where(c, m.chosen.data, m.fallback.data),
                (self.MASK, ts.Tensor([[0, 1], [1, 0]], dtype=ts.uint8)),
            ),
        ):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                for value in inputs:
                    replayed = model(value).data
                    expected = forward(model, value)
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_gradients_reach_the_differentiable_parameters(self):
        for factory, inputs in (
            (self.Rectified, self.INPUTS),
            (self.Bounded, self.INPUTS),
            (self.Selected, self.MASK),
        ):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                ts.backward(ts.sum(model(inputs)))
                parameters = model.parameters()
                self.assertTrue(parameters)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.shape, parameter.shape)

    def test_a_recorded_condition_receives_no_gradient(self):
        # ``where`` refuses a differentiable condition and its rule never
        # propagates through one. Recording the condition as a vertex does
        # not change that.
        model = self.Selected()

        ts.backward(ts.sum(model(self.MASK)))

        condition_leaf = model._structure.inputs[0]
        self.assertTrue(condition_leaf.is_bound)
        self.assertFalse(condition_leaf.variable.requires_grad)
        self.assertIsNone(condition_leaf.variable.grad)
        # Both value branches did receive one.
        self.assertIsNotNone(model.chosen.grad)
        self.assertIsNotNone(model.fallback.grad)

    def test_replay_reuses_the_output_variable(self):
        model = self.Rectified()

        first = model(self.INPUTS)
        second = model(ts.Tensor([[-5.0, -6.0], [7.0, -8.0]]))

        self.assertIs(first, second)

    def test_each_migrated_function_builds_a_model_of_its_own(self):
        cases = (
            ("maximum", lambda self, x: ts.maximum(x @ self.w, self.floor)),
            ("minimum", lambda self, x: ts.minimum(x @ self.w, self.floor)),
            ("clip", lambda self, x: ts.clip(x @ self.w, -1.0, 1.0)),
            ("where", lambda self, x: ts.where(
                ts.Tensor([[1, 0], [0, 1]], dtype=ts.uint8),
                x @ self.w,
                self.floor,
            )),
        )
        for name, forward in cases:
            with self.subTest(function=name):
                reset_graph_state()

                def __init__(self):
                    ts.Graph.__init__(self)
                    self.w = ts.Variable(
                        [[1.0, -0.5], [-0.5, 1.0]], name="w"
                    )
                    self.floor = ts.Variable(
                        [[0.0, 0.0], [0.0, 0.0]], name="floor"
                    )

                model = type(
                    "Single",
                    (ts.Graph,),
                    {"__init__": __init__, "forward": forward},
                )()

                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                names = [
                    instruction.operation.name
                    for instruction in program._instructions
                ]
                self.assertEqual(names[-1], name)
                self.assertIsInstance(model(self.INPUTS), ts.Variable)



class ConvolutionModelConstructionTests(unittest.TestCase):
    """A convolution with trainable weights builds the model's program.

    Constructing these models used to raise a plain TypeError: a Variable
    kernel took the convolution's own wrapping branch, which handed the
    input vertex to Tensor construction, and that is not the signal the
    build treats as a fallback.
    """

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Signal(ts.Graph):
        def __init__(self):
            super().__init__()
            self.kernel = ts.Variable([[[1.0, -1.0]]], name="kernel")
            self.bias = ts.Variable([0.5], name="bias")

        def forward(self, x):
            return ts.conv1d(x, self.kernel, self.bias)

    class Image(ts.Graph):
        def __init__(self):
            super().__init__()
            self.kernel = ts.Variable(
                [[[[1.0, 0.0], [0.0, -1.0]]]], name="kernel"
            )

        def forward(self, x):
            return ts.conv2d(x, self.kernel)

    class Volume(ts.Graph):
        def __init__(self):
            super().__init__()
            self.kernel = ts.Variable(
                [[[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]],
                name="kernel",
            )

        def forward(self, x):
            return ts.conv3d(x, self.kernel)

    SIGNALS = (
        ts.Tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        ts.Tensor([[[-1.0, 0.5, 2.0, -3.0]]]),
    )
    IMAGES = (
        ts.Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
        ts.Tensor([[[[0.5, -1.0, 2.0], [3.0, 0.0, -2.0], [1.0, 1.0, 1.0]]]]),
    )
    VOLUMES = (
        ts.Tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]]),
        ts.Tensor([[[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]]]),
    )

    def _cases(self):
        return (
            (self.Signal, "conv1d", self.SIGNALS,
             lambda m, x: ts.conv1d(x, m.kernel.data, m.bias.data)),
            (self.Image, "conv2d", self.IMAGES,
             lambda m, x: ts.conv2d(x, m.kernel.data)),
            (self.Volume, "conv3d", self.VOLUMES,
             lambda m, x: ts.conv3d(x, m.kernel.data)),
        )

    def test_a_convolution_model_builds_during_construction(self):
        for factory, name, _, _ in self._cases():
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()

                self.assertIsNotNone(model._structure)
                program = model._structure.computations[0]
                self.assertEqual(
                    [
                        instruction.operation.name
                        for instruction in program._instructions
                    ],
                    [name],
                )

    def test_the_built_programs_replay_what_eager_calculates(self):
        for factory, _, inputs, forward in self._cases():
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                for value in inputs:
                    replayed = model(value).data
                    expected = forward(model, value)
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_gradients_reach_the_kernel_and_bias(self):
        for factory, _, inputs, _ in self._cases():
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                ts.backward(ts.sum(model(inputs[0])))
                parameters = model.parameters()
                self.assertTrue(parameters)
                for parameter in parameters:
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.shape, parameter.shape)

    def test_replay_reuses_the_output_variable(self):
        model = self.Signal()

        first = model(self.SIGNALS[0])
        second = model(self.SIGNALS[1])

        self.assertIs(first, second)

    def test_a_real_type_error_in_forward_still_propagates(self):
        class Broken(ts.Graph):
            def __init__(self):
                super().__init__()
                self.kernel = ts.Variable([[[1.0, 1.0]]], name="kernel")

            def forward(self, x):
                raise TypeError("a real bug in forward")

        with self.assertRaisesRegex(TypeError, "a real bug in forward"):
            Broken()



class LossModelConstructionTests(unittest.TestCase):
    """A binary cross-entropy head builds; a cross-entropy head still traces."""

    INPUTS = (
        ts.Tensor([[1.0, 2.0]]),
        ts.Tensor([[-2.0, 0.5]]),
        ts.Tensor([[3.0, -1.0]]),
    )

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    class Classifier(ts.Graph):
        """dot -> sigmoid -> binary_cross_entropy against a fixed target."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0], [0.5]], name="w")

        def forward(self, x):
            return ts.binary_cross_entropy(
                ts.sigmoid(x @ self.w), ts.Tensor([[1.0]])
            )

    class Logits(ts.Graph):
        """The stable form, taking the logits directly."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0], [0.5]], name="w")

        def forward(self, x):
            return ts.binary_cross_entropy(
                x @ self.w, ts.Tensor([[1.0]]), from_logits=True
            )

    class Trainable(ts.Graph):
        """A target that is itself a parameter, so it takes gradients."""

        def __init__(self):
            super().__init__()
            self.w = ts.Variable([[1.0], [0.5]], name="w")
            self.target = ts.Variable([[0.75]], name="target")

        def forward(self, x):
            return ts.binary_cross_entropy(
                ts.sigmoid(x @ self.w), self.target
            )

    def test_the_models_build_during_construction(self):
        expected = {
            "Classifier": ["dot", "sigmoid", "binary_cross_entropy"],
            "Logits": ["dot", "binary_cross_entropy"],
            "Trainable": ["dot", "sigmoid", "binary_cross_entropy"],
        }
        for model in (self.Classifier(), self.Logits(), self.Trainable()):
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
            (
                self.Classifier,
                lambda m, x: ts.binary_cross_entropy(
                    ts.sigmoid(x @ m.w.data), ts.Tensor([[1.0]])
                ),
            ),
            (
                self.Logits,
                lambda m, x: ts.binary_cross_entropy(
                    x @ m.w.data, ts.Tensor([[1.0]]), from_logits=True
                ),
            ),
            (
                self.Trainable,
                lambda m, x: ts.binary_cross_entropy(
                    ts.sigmoid(x @ m.w.data), m.target.data
                ),
            ),
        ):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                for value in self.INPUTS:
                    replayed = model(value).data
                    expected = forward(model, value)
                    self.assertEqual(replayed.tolist(), expected.tolist())
                    self.assertEqual(replayed.shape, expected.shape)

    def test_gradients_reach_the_prediction_parameters(self):
        for factory in (self.Classifier, self.Logits):
            with self.subTest(model=factory.__name__):
                reset_graph_state()
                model = factory()
                ts.backward(model(self.INPUTS[0]))
                self.assertIsNotNone(model.w.grad)
                self.assertEqual(model.w.grad.shape, model.w.shape)

    def test_a_trainable_target_still_takes_a_gradient(self):
        # The target's differentiability is the operation's own rule, and
        # recording it structurally does not change which side receives one.
        model = self.Trainable()

        ts.backward(model(self.INPUTS[0]))

        self.assertIsNotNone(model.w.grad)
        self.assertIsNotNone(model.target.grad)
        self.assertEqual(model.target.grad.shape, model.target.shape)

    def test_a_constant_target_takes_no_gradient(self):
        model = self.Classifier()

        ts.backward(model(self.INPUTS[0]))

        target_leaf = model._structure.computations[0]._variable_nodes
        constants = [
            node.variable
            for node in target_leaf
            if node.is_bound and not node.variable.requires_grad
        ]
        self.assertTrue(constants)
        for constant in constants:
            self.assertIsNone(constant.grad)

    def test_replay_reuses_the_output_variable(self):
        model = self.Classifier()

        first = model(self.INPUTS[0])
        second = model(self.INPUTS[1])

        self.assertIs(first, second)

    def test_a_cross_entropy_head_still_keeps_tracing(self):
        # cross_entropy prepares its operands from the logits' shape and
        # values, so it cannot be recorded and the model stays on tracing.
        class Multiclass(ts.Graph):
            def __init__(self):
                super().__init__()
                self.w = ts.Variable([[1.0, 0.5], [0.5, 1.0]], name="w")

            def forward(self, x):
                return ts.cross_entropy(
                    x @ self.w, ts.Tensor([0], dtype=ts.int64)
                )

        model = Multiclass()

        self.assertIsNone(model._structure)
        # Still usable, and still differentiable, through tracing.
        loss = model(ts.Tensor([[1.0, 2.0]]))
        self.assertIsInstance(loss, ts.Variable)
        ts.backward(loss)
        self.assertIsNotNone(model.w.grad)


if __name__ == "__main__":
    unittest.main()
