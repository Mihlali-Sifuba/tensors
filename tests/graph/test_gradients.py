import ast
import inspect
import subprocess
import sys
import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph import computation as computation_package
from tensors.graph.computation import computation as computation_module
from tensors.graph.computation import fusion, gradients
from tensors.graph.state import reset_graph_state


class GradientModuleBoundaryTests(unittest.TestCase):
    """The generic gradient mechanics live beside the execution model."""

    HELPERS = (
        "gradient_seed",
        "sum_gradient_values",
        "sum_gradient_graph",
        "validate_gradients",
    )

    def test_helpers_are_no_longer_methods_on_computation(self):
        for name in (
            "_gradient_seed",
            "_sum_gradient_values",
            "_sum_gradient_graph",
            "_validate_gradients",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(Computation, name))
        # No forwarding wrappers under the new names either.
        for name in self.HELPERS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(Computation, name))

    def test_gradients_module_owns_the_helpers(self):
        for name in self.HELPERS:
            with self.subTest(name=name):
                function = getattr(gradients, name)
                self.assertEqual(
                    function.__module__,
                    "tensors.graph.computation.gradients",
                )

    def test_gradients_module_defines_no_classes(self):
        classes = [
            name
            for name, value in vars(gradients).items()
            if isinstance(value, type) and value.__module__ == gradients.__name__
        ]
        self.assertEqual(classes, [])

    def test_gradients_module_does_not_depend_on_computation(self):
        # The dependency runs one way: computation and fusion both use the
        # gradient mechanics; the mechanics know nothing about either.
        self.assertNotIn("Computation", vars(gradients))
        for module in ("computation", "fusion", "autograd"):
            with self.subTest(module=module):
                self.assertNotIn(module, self._imports_of(gradients))

    def test_fusion_no_longer_imports_computation_at_runtime(self):
        self.assertNotIn("Computation", vars(fusion))
        computation_imports = [
            alias.name
            for node in ast.walk(ast.parse(inspect.getsource(fusion)))
            if isinstance(node, ast.ImportFrom) and node.module == "computation"
            for alias in node.names
        ]
        self.assertEqual(computation_imports, [])

    def test_both_execution_paths_share_one_accumulation_rule(self):
        self.assertIs(
            computation_module.sum_gradient_values,
            gradients.sum_gradient_values,
        )
        self.assertIs(fusion.sum_gradient_values, gradients.sum_gradient_values)

    def test_helpers_are_not_public_api(self):
        for name in self.HELPERS:
            with self.subTest(name=name):
                self.assertNotIn(name, computation_package.__all__)
                self.assertNotIn(name, ts.graph.__all__)
                self.assertFalse(hasattr(computation_package, name))
                self.assertFalse(hasattr(ts.graph, name))
                self.assertFalse(hasattr(ts, name))

    def test_module_imports_first_without_a_cycle(self):
        for name in ("gradients", "fusion", "computation", "autograd"):
            with self.subTest(module=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"import tensors.graph.computation.{name}",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    @staticmethod
    def _imports_of(module):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(alias.name for alias in node.names)
        return imported


class GradientHelperBehaviourTests(unittest.TestCase):
    """The extracted mechanics behave exactly as the methods did."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_seed_defaults_to_ones_in_the_output_dtype(self):
        value = ts.Variable(ts.Tensor([2.0, 3.0], dtype=ts.float32))
        output = value * value

        seed = gradients.gradient_seed(output, None)

        self.assertIsInstance(seed, ts.Tensor)
        self.assertEqual(seed.tolist(), [1.0, 1.0])
        self.assertIs(seed.dtype, ts.float32)

    def test_seed_accepts_a_scalar_for_a_scalar_output(self):
        value = ts.Variable(ts.Tensor([2.0], shape=()))
        output = value * value

        seed = gradients.gradient_seed(output, 3.0)

        self.assertEqual(seed.shape, ())
        self.assertEqual(seed.tolist(), [3.0])

    def test_seed_casts_to_the_output_dtype(self):
        value = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        output = value * value

        seed = gradients.gradient_seed(
            output, ts.Tensor([3.0], dtype=ts.float64)
        )

        self.assertIs(seed.dtype, ts.float32)

    def test_seed_rejects_a_mismatched_shape(self):
        value = ts.Variable([2.0, 3.0])
        output = value * value

        with self.assertRaisesRegex(ValueError, "does not match output shape"):
            gradients.gradient_seed(output, ts.Tensor([1.0]))

    def test_seed_returns_a_variable_for_create_graph(self):
        value = ts.Variable([2.0], requires_grad=True)
        output = value * value

        seed = gradients.gradient_seed(output, None, create_graph=True)

        self.assertIsInstance(seed, ts.Variable)
        self.assertFalse(seed.requires_grad)

    def test_seed_rejects_a_differentiable_seed_of_another_dtype(self):
        value = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        output = value * value
        seed = ts.Variable(
            ts.Tensor([1.0], dtype=ts.float64), requires_grad=True
        )

        with self.assertRaisesRegex(TypeError, "same dtype as the output"):
            gradients.gradient_seed(output, seed, create_graph=True)

    def test_summing_one_contribution_returns_it_unchanged(self):
        only = ts.Tensor([1.0, 2.0])

        self.assertIs(gradients.sum_gradient_values([only]), only)

    def test_summing_preserves_dtype_and_shape(self):
        left = ts.Tensor([1.0, 2.0], dtype=ts.float32)
        right = ts.Tensor([0.5, 0.25], dtype=ts.float32)

        total = gradients.sum_gradient_values([left, right])

        self.assertEqual(total.tolist(), [1.5, 2.25])
        self.assertIs(total.dtype, ts.float32)
        self.assertEqual(total.shape, (2,))

    def test_summing_preserves_a_multidimensional_shape(self):
        left = ts.full((2, 3), 0.25)
        right = ts.full((2, 3), 0.5)

        total = gradients.sum_gradient_values([left, right])

        self.assertEqual(total.shape, (2, 3))
        self.assertEqual(total.size, 6)
        self.assertEqual(
            [float(total._data[index]) for index in range(total.size)],
            [0.75] * 6,
        )

    def test_summing_matches_across_backends(self):
        expected = None
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                terms = [
                    ts.full((64,), 0.25),
                    ts.full((64,), 0.5),
                    ts.full((64,), 0.125),
                ]
                total = gradients.sum_gradient_values(terms).tolist()
                if expected is None:
                    expected = total
                self.assertEqual(total, expected)

    def test_graph_summation_stays_differentiable(self):
        value = ts.Variable([2.0], requires_grad=True)
        left = value * 3.0
        right = value * 5.0

        total = gradients.sum_gradient_graph([left, right])

        self.assertIsInstance(total, ts.Variable)
        self.assertEqual(total.data.tolist(), [16.0])
        self.assertEqual(ts.grad(ts.sum(total), value).tolist(), [8.0])

    def test_validation_enforces_the_demand_contract(self):
        left = ts.Variable([2.0], requires_grad=True)
        right = ts.Variable([3.0], requires_grad=True)
        operation = (left * right).node.producer.operation
        inputs = (left, right)

        with self.assertRaisesRegex(RuntimeError, "did not request"):
            gradients.validate_gradients(
                operation,
                inputs,
                (ts.Tensor([1.0]), ts.Tensor([1.0])),
                (True, False),
                graph=False,
            )
        with self.assertRaisesRegex(RuntimeError, "returned None for input"):
            gradients.validate_gradients(
                operation,
                inputs,
                (None, None),
                (True, False),
                graph=False,
            )
        with self.assertRaisesRegex(RuntimeError, "gradients for"):
            gradients.validate_gradients(
                operation,
                inputs,
                (ts.Tensor([1.0]),),
                (True, False),
                graph=False,
            )
        with self.assertRaisesRegex(TypeError, "must be a Tensor"):
            gradients.validate_gradients(
                operation,
                inputs,
                (left, None),
                (True, False),
                graph=False,
            )
        with self.assertRaisesRegex(ValueError, "expected"):
            gradients.validate_gradients(
                operation,
                inputs,
                (ts.Tensor([1.0, 2.0]), None),
                (True, False),
                graph=False,
            )

    def test_validation_keeps_a_requested_zero(self):
        value = ts.Variable([2.0], requires_grad=True)
        other = ts.Variable([3.0], requires_grad=True)
        operation = (value * other).node.producer.operation
        zero = ts.Tensor([0.0])

        validated = gradients.validate_gradients(
            operation,
            (value, other),
            (zero, None),
            (True, False),
            graph=False,
        )

        self.assertIs(validated[0], zero)
        self.assertIsNone(validated[1])

    def test_validation_restores_the_input_dtype(self):
        value = ts.Variable(
            ts.Tensor([2.0], dtype=ts.float32), requires_grad=True
        )
        other = ts.Variable(
            ts.Tensor([3.0], dtype=ts.float32), requires_grad=True
        )
        operation = (value * other).node.producer.operation

        validated = gradients.validate_gradients(
            operation,
            (value, other),
            (ts.Tensor([1.0], dtype=ts.float64), None),
            (True, False),
            graph=False,
        )

        self.assertIs(validated[0].dtype, ts.float32)


class SharedAccumulationTests(unittest.TestCase):
    """Fused and ordinary reverse passes accumulate gradients alike."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_branching_contributions_agree_with_and_without_fusion(self):
        from unittest.mock import patch

        def build():
            value = ts.Variable(ts.full((256,), 0.5), requires_grad=True)
            shared = ts.sin(value * 1.5) + 0.25
            return ts.sum(shared * shared + ts.tanh(shared)), value

        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                reset_graph_state()
                output, value = build()
                Computation(output).backward()
                fused = value.grad.tolist()

                reset_graph_state()
                with patch.object(
                    computation_module,
                    "execute_fused_backward",
                    return_value=False,
                ):
                    output, value = build()
                    Computation(output).backward()
                    unfused = value.grad.tolist()

                self.assertEqual(fused, unfused)


if __name__ == "__main__":
    unittest.main()
