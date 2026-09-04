import ast
import inspect
import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph import autograd
from tensors.graph import computation as computation_module
from tensors.graph.autograd import computation_for
from tensors.graph.state import reset_graph_state


class FunctionalApiBoundaryTests(unittest.TestCase):
    """The functional API lives beside the execution model, not inside it."""

    def test_computation_module_defines_no_functional_api(self):
        for name in ("backward", "grad", "computation_for"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(computation_module, name))
        self.assertEqual(computation_module.__all__, ["Computation"])

    def test_functional_api_is_defined_by_the_autograd_module(self):
        for function in (ts.backward, ts.grad):
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__module__, "tensors.graph.autograd")
        self.assertIs(ts.backward, autograd.backward)
        self.assertIs(ts.grad, autograd.grad)
        self.assertIs(ts.graph.backward, autograd.backward)
        self.assertIs(ts.graph.grad, autograd.grad)

    def test_computation_no_longer_resolves_its_own_autograd_plan(self):
        self.assertFalse(hasattr(Computation, "_for_autograd"))

    def test_reverse_execution_stays_on_the_computation(self):
        for name in (
            "backward",
            "_live_slots",
            "_gradient_seed",
            "_backward_values",
            "_backward_graph",
            "_execute_backward_instruction",
            "_validate_recorded_states",
            "_sum_gradient_values",
            "_sum_gradient_graph",
            "_validate_gradients",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(Computation, name))

    def test_computation_does_not_import_the_autograd_module(self):
        # The dependency runs one way: autograd -> computation -> fusion.
        source = inspect.getsource(computation_module)
        imported: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        self.assertNotIn("autograd", imported)
        self.assertNotIn(".autograd", imported)
        self.assertNotIn("tensors.graph.autograd", imported)

    def test_computation_for_is_not_public_api(self):
        self.assertNotIn("computation_for", ts.graph.__all__)
        self.assertNotIn("computation_for", autograd.__all__)
        self.assertFalse(hasattr(ts.graph, "computation_for"))
        self.assertFalse(hasattr(ts, "computation_for"))

    def test_public_names_survive_the_move(self):
        for name in ("backward", "grad", "Computation"):
            with self.subTest(name=name):
                self.assertIn(name, ts.graph.__all__)
                self.assertTrue(hasattr(ts.graph, name))
        self.assertIn("backward", ts.__all__)
        self.assertIn("grad", ts.__all__)


class ComputationForTests(unittest.TestCase):
    """The cached reverse plan is resolved by a module function."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_creates_and_caches_a_plan_when_none_exists(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)
        self.assertIsNone(output._autograd_computation)

        computation = computation_for(output)

        self.assertIsInstance(computation, Computation)
        self.assertIs(output._autograd_computation, computation)
        self.assertIs(computation.output, output)

    def test_reuses_an_active_cached_plan(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        first = computation_for(output)
        second = computation_for(output)

        self.assertIs(second, first)
        # The public entry points resolve the same cached plan.
        ts.backward(output)
        self.assertIs(output._autograd_computation, first)
        ts.grad(output, value)
        self.assertIs(output._autograd_computation, first)

    def test_replaces_a_released_cached_plan(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        first = computation_for(output)
        first.release()
        second = computation_for(output)

        self.assertIsNot(second, first)
        self.assertIs(output._autograd_computation, second)
        self.assertEqual(second.forward().tolist(), [4.0])

    def test_differentiating_through_a_released_plan_succeeds(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        computation_for(output).release()
        ts.backward(output)

        self.assertEqual(value.grad.tolist(), [4.0])

    def test_rejects_an_output_without_a_graph_node(self):
        with self.assertRaisesRegex(TypeError, "graph node"):
            computation_for(ts.Tensor([1.0]))


class FunctionalApiBehaviourTests(unittest.TestCase):
    """The moved functions behave exactly as they did before."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_backward_publishes_gradients(self):
        value = ts.Variable([2.0, 3.0], requires_grad=True)
        output = ts.sum(value * value)

        ts.backward(output)

        self.assertEqual(value.grad.tolist(), [4.0, 6.0])

    def test_backward_accepts_an_explicit_seed(self):
        value = ts.Variable([2.0, 3.0], requires_grad=True)
        output = value * value

        ts.backward(output, ts.Tensor([10.0, 20.0]))

        self.assertEqual(value.grad.tolist(), [40.0, 120.0])

    def test_backward_rejects_a_non_boolean_create_graph(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        with self.assertRaisesRegex(TypeError, "create_graph must be a bool"):
            ts.backward(output, create_graph=1)

    def test_grad_returns_one_gradient_for_a_single_variable(self):
        value = ts.Variable([2.0, 3.0], requires_grad=True)
        output = ts.sum(value * value)

        result = ts.grad(output, value)

        self.assertIsInstance(result, ts.Tensor)
        self.assertEqual(result.tolist(), [4.0, 6.0])

    def test_grad_returns_a_tuple_for_an_iterable_of_variables(self):
        left = ts.Variable([2.0], requires_grad=True)
        right = ts.Variable([3.0], requires_grad=True)
        output = ts.sum(left * right)

        result = ts.grad(output, (left, right))

        self.assertIsInstance(result, tuple)
        self.assertEqual([gradient.tolist() for gradient in result],
                         [[3.0], [2.0]])
        # A one-element iterable still returns a tuple.
        self.assertIsInstance(ts.grad(output, [left]), tuple)

    def test_grad_does_not_mutate_published_gradients(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        ts.grad(output, value)

        self.assertIsNone(value.grad)

    def test_grad_honours_grad_outputs(self):
        value = ts.Variable([2.0, 3.0], requires_grad=True)
        output = value * value

        result = ts.grad(output, value, ts.Tensor([10.0, 20.0]))

        self.assertEqual(result.tolist(), [40.0, 120.0])

    def test_grad_builds_a_differentiable_graph_on_request(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value ** 3.0)

        first = ts.grad(output, value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertIsInstance(first, ts.Variable)
        self.assertEqual(first.data.tolist(), [12.0])
        self.assertEqual(second.tolist(), [12.0])

    def test_grad_prunes_paths_it_was_not_asked_for(self):
        value = ts.Variable([2.0], requires_grad=True)
        other = ts.Variable([5.0], requires_grad=True)
        output = ts.sum(value * value + other)

        result = ts.grad(output, value)

        self.assertEqual(result.tolist(), [4.0])
        self.assertIsNone(other.grad)

    def test_grad_validates_its_inputs(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)

        with self.assertRaisesRegex(ValueError, "at least one input"):
            ts.grad(output, ())
        with self.assertRaisesRegex(TypeError, "must be a Variable"):
            ts.grad(output, (ts.Tensor([1.0]),))
        with self.assertRaisesRegex(TypeError, "Variable or an iterable"):
            ts.grad(output, 3.0)
        with self.assertRaisesRegex(TypeError, "create_graph must be a bool"):
            ts.grad(output, value, create_graph=None)

    def test_mutation_after_the_forward_pass_is_rejected(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = ts.sum(value * value)
        value.data[0] = 5.0

        with self.assertRaisesRegex(RuntimeError, "modified after its"):
            ts.grad(output, value)
        with self.assertRaisesRegex(RuntimeError, "modified after its"):
            ts.backward(output)


if __name__ == "__main__":
    unittest.main()
