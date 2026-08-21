"""Regression coverage for the package-wide correctness audit."""

from functools import partial
import math
import sys
import unittest

import tensors as ts
from tensors.dtype import DataType
from tensors.graph import Computation


class AutogradRegressionTests(unittest.TestCase):
    def test_frozen_root_does_not_receive_a_gradient(self):
        value = ts.Variable([2.0], requires_grad=False)
        value.grad = ts.Tensor([9.0])

        self.assertIsNone(ts.grad(value, value))
        ts.backward(value)

        self.assertIsNone(value.grad)

    def test_python_seed_matches_a_zero_dimensional_output(self):
        value = ts.Variable([3.0, 4.0])
        output = ts.dot(value, value)

        derivative = ts.grad(output, value, grad_outputs=2.0)

        self.assertEqual(output.shape, ())
        self.assertEqual(derivative.tolist(), [12.0, 16.0])

    def test_division_vjp_avoids_square_overflow_and_underflow(self):
        left = ts.Variable([1.0e308])
        right = ts.Variable([1.0e308])
        derivative = ts.grad(
            left / right,
            right,
            grad_outputs=ts.Tensor([1.0e308]),
        )
        self.assertEqual(derivative.tolist(), [-1.0])

        denominator = ts.Variable([1.0e-300])
        derivative = ts.grad(1.0 / denominator, denominator)
        self.assertEqual(derivative.tolist(), [-math.inf])

    def test_broadcast_product_vjp_preserves_exact_cancellation(self):
        value = ts.Variable([0.0])
        factor = ts.Variable([1.0e308, -1.0e308], requires_grad=False)
        seed = ts.Variable([2.0, 2.0])

        derivative = ts.grad(
            value * factor,
            value,
            grad_outputs=seed,
            create_graph=True,
        )

        self.assertEqual(derivative.data.tolist(), [0.0])
        self.assertEqual(ts.grad(derivative, seed).tolist(), [1.0e308, -1.0e308])

    def test_logsumexp_infinity_subgradient_retains_seed_connectivity(self):
        value = ts.Variable([math.inf, math.inf])
        seed = ts.Variable([2.0])

        derivative = ts.grad(
            ts.logsumexp(value),
            value,
            grad_outputs=seed,
            create_graph=True,
        )

        self.assertEqual(derivative.data.tolist(), [1.0, 1.0])
        self.assertEqual(
            ts.grad(derivative, seed, ts.Tensor([1.0, 1.0])).tolist(),
            [1.0],
        )

    def test_custom_scalar_metadata_is_not_treated_as_operator_dispatch(self):
        class MetadataOperation:
            @staticmethod
            def forward(value, *, scalar):
                return value + scalar

            @staticmethod
            def backward(gradient, *inputs, **kwargs):
                return [gradient]

            @staticmethod
            def backward_graph(gradient, *inputs, **kwargs):
                return [gradient]

        value = ts.Variable([1.0])
        output = ts.Variable._from_operation(
            MetadataOperation.forward(value.data, scalar=2.0),
            "metadata",
            MetadataOperation,
            [value],
            scalar=2.0,
        )
        value.data = ts.Tensor([4.0])

        replayed = Computation(output).forward()

        self.assertEqual(replayed.tolist(), [6.0])


class NumericalRegressionTests(unittest.TestCase):
    def test_softmax_family_preserves_small_normalization_tails(self):
        self.assertEqual(
            ts.logsumexp(ts.Tensor([0.0, -40.0])).item(),
            math.log1p(math.exp(-40.0)),
        )
        log_probabilities = ts.log_softmax(
            ts.Tensor([1.0e16, 1.0e16 - 2.0])
        )
        self.assertAlmostEqual(log_probabilities._data[0], -0.1269280110429725)
        self.assertAlmostEqual(log_probabilities._data[1], -2.1269280110429727)

        probabilities = ts.softmax(ts.Tensor([0.0, -37.0]))
        self.assertLess(probabilities._data[0], 1.0)
        self.assertGreater(probabilities._data[1], 0.0)

    def test_cross_entropy_uses_stable_group_sums_and_mean(self):
        maximum = sys.float_info.max
        overflowing = ts.cross_entropy(
            ts.Tensor([0.0, -maximum, -maximum]),
            ts.Tensor([0.0, 0.500000025, 0.500000025]),
        )
        self.assertEqual(overflowing.item(), math.inf)

        tiny = math.ulp(0.0)
        count = 10_000
        logits = ts.Tensor([0.0, -745.0] * count, shape=(count, 2))
        targets = ts.Tensor([1.0, tiny] * count, shape=(count, 2))
        self.assertGreater(ts.cross_entropy(logits, targets).item(), 0.0)

    def test_binary_cross_entropy_logits_avoids_large_value_cancellation(self):
        target = math.nextafter(1.0, 0.0)
        loss = ts.binary_cross_entropy(
            ts.Tensor([1.0e16]),
            ts.Tensor([target]),
            from_logits=True,
        )
        self.assertEqual(loss.item(), (1.0 - target) * 1.0e16)

    def test_saturated_activation_and_loss_derivatives_remain_nonzero(self):
        sigmoid_input = ts.Variable([40.0])
        self.assertEqual(
            ts.grad(ts.sigmoid(sigmoid_input), sigmoid_input).item(),
            4.248354255291589e-18,
        )

        tanh_input = ts.Variable([20.0])
        self.assertEqual(
            ts.grad(ts.tanh(tanh_input), tanh_input).item(),
            1.6993417021166355e-17,
        )

        logit = ts.Variable([40.0])
        loss = ts.binary_cross_entropy(
            logit,
            ts.Tensor([1.0]),
            from_logits=True,
        )
        self.assertEqual(ts.grad(loss, logit).item(), -4.248354255291589e-18)

        logits = ts.Variable([0.0, -40.0])
        loss = ts.cross_entropy(logits, ts.Tensor([0], dtype=ts.int64))
        self.assertEqual(
            ts.grad(loss, logits).tolist(),
            [-4.248354255291589e-18, 4.248354255291589e-18],
        )

    def test_empty_means_are_nan(self):
        self.assertTrue(math.isnan(ts.mean(ts.Tensor([])).item()))
        self.assertTrue(math.isnan(ts.std(ts.Tensor([])).item()))
        self.assertTrue(math.isnan(ts.binary_cross_entropy([], []).item()))
        self.assertTrue(math.isnan(ts.cross_entropy(
            ts.Tensor([], shape=(0, 2)),
            ts.Tensor([], dtype=ts.int64),
        ).item()))


class OptimizerRegressionTests(unittest.TestCase):
    def test_adaptive_optimizers_handle_huge_finite_gradients(self):
        for optimizer_type, learning_rate in (
            (ts.optim.Adam, 0.1),
            (ts.optim.RMSprop, 0.01),
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                parameter = ts.Variable([1.0])
                parameter.grad = ts.Tensor([1.0e308])
                optimizer = optimizer_type(
                    [parameter], learning_rate=learning_rate
                )

                optimizer.step()

                self.assertTrue(math.isfinite(parameter.data.item()))
                self.assertAlmostEqual(parameter.data.item(), 0.9)

    def test_adam_bias_correction_tracks_mutated_betas(self):
        parameter = ts.Variable([1.0])
        optimizer = ts.optim.Adam(
            [parameter],
            learning_rate=0.1,
            betas=(0.5, 0.0),
            eps=1.0e-12,
        )
        parameter.grad = ts.Tensor([1.0])
        optimizer.step()
        optimizer.beta1 = 0.8
        parameter.grad = ts.Tensor([1.0])

        optimizer.step()

        self.assertAlmostEqual(parameter.data.item(), 0.8, places=10)

    def test_optimizer_steps_are_atomic_after_gradient_validation(self):
        for optimizer_type in (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                first = ts.Variable([1.0])
                second = ts.Variable([2.0])
                first.grad = ts.Tensor([1.0])
                second.grad = ts.Tensor([1.0, 2.0])
                optimizer = optimizer_type(
                    [first, second], learning_rate=0.1
                )

                with self.assertRaises(ValueError):
                    optimizer.step()

                self.assertEqual(first.data.tolist(), [1.0])
                self.assertEqual(second.data.tolist(), [2.0])


class DiscoveryAndTensorRegressionTests(unittest.TestCase):
    def test_parameter_discovery_traverses_partial_decorated_and_slotted_callables(self):
        first = ts.Variable([1.0])

        def multiply(value, *, weight):
            return value * weight

        self.assertEqual(ts.Graph(partial(multiply, weight=first)).parameters(), [first])

        second = ts.Variable([2.0])

        class SlottedCallable:
            __slots__ = ("weight",)

            def __init__(self, weight):
                self.weight = weight

            def __call__(self, value):
                return value * self.weight

        self.assertEqual(ts.Graph(SlottedCallable(second)).parameters(), [second])

        third = ts.Variable([3.0])

        def original(value):
            return value * third

        def decorated(*args, **kwargs):
            return original(*args, **kwargs)

        self.assertEqual(ts.Graph(decorated).parameters(), [third])

    def test_integer_conversion_is_consistent_across_entry_points(self):
        constructed = ts.Tensor([1.9], dtype=ts.int32)
        copied = ts.Tensor(ts.Tensor([1.9]), dtype=ts.int32)
        assigned = ts.Tensor([0], dtype=ts.int32)
        sliced = ts.Tensor([0], dtype=ts.int32)
        assigned[0] = 1.9
        sliced[:] = [1.9]

        self.assertEqual(constructed.tolist(), [1])
        self.assertEqual(copied.tolist(), [1])
        self.assertEqual(assigned.tolist(), [1])
        self.assertEqual(sliced.tolist(), [1])

    def test_boolean_indices_are_rejected_consistently(self):
        tensor = ts.Tensor([1.0, 2.0])
        variable = ts.Variable([1.0, 2.0])

        for action in (
            lambda: tensor[True],
            lambda: tensor.__setitem__(False, 3.0),
            lambda: tensor[(True,)],
            lambda: variable[True],
        ):
            with self.subTest(action=action):
                with self.assertRaises(TypeError):
                    action()

    def test_deep_rank_data_and_slices_do_not_use_python_recursion(self):
        data = 1.0
        depth = 1_050
        for _ in range(depth):
            data = [data]

        tensor = ts.Tensor(data)
        sliced = tensor[(slice(None),) * depth]

        self.assertEqual(tensor.ndim, depth)
        self.assertEqual(sliced.shape, tensor.shape)
        self.assertIn("shape=", repr(tensor))

    def test_cyclic_nested_data_is_rejected(self):
        data = []
        data.append(data)

        with self.assertRaisesRegex(ValueError, "Cyclic"):
            ts.Tensor(data)


class ValidationRegressionTests(unittest.TestCase):
    def test_gradcheck_rejects_nonfinite_inputs(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    ts.gradcheck(lambda item: item * item, ts.Tensor([value]))

    def test_boolean_options_and_numeric_hyperparameters_are_strict(self):
        value = ts.Variable([1.0])
        output = value * value
        actions = (
            lambda: ts.sum(ts.Tensor([1.0]), keepdims=1),
            lambda: ts.math.Softmax.forward(ts.Tensor([1.0]), keepdims="yes"),
            lambda: ts.math.Concat.forward(ts.Tensor([1.0]), keepdims=1),
            lambda: ts.grad(output, value, create_graph=1),
            lambda: ts.jacobian(output, value, create_graph="yes"),
            lambda: ts.gradcheck(
                lambda item: item * item,
                ts.Tensor([1.0]),
                raise_exception=1,
            ),
            lambda: ts.optim.SGD([], learning_rate=True),
            lambda: ts.optim.Adam([], betas=(True, 0.9)),
            lambda: ts.optim.RMSprop([], rho=False),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(TypeError):
                    action()

    def test_custom_dtype_metadata_must_match_array_storage(self):
        with self.assertRaises(ValueError):
            DataType("fake64", "d", 1)
        with self.assertRaises(ValueError):
            DataType("unsupported", "z", 8)
        with self.assertRaises(ValueError):
            DataType("", "d", 8)
        with self.assertRaises(TypeError):
            DataType("fake64", "d", True)


if __name__ == "__main__":
    unittest.main()
