import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class StableLossTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def assertValuesAlmostEqual(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_logsumexp_is_stable_for_extreme_finite_values(self):
        result = ts.logsumexp(ts.Tensor([1000.0, 1001.0]))
        expected = 1001.0 + math.log1p(math.exp(-1.0))

        self.assertTrue(math.isfinite(result.item()))
        self.assertAlmostEqual(result.item(), expected)

    def test_logsumexp_supports_multiple_axes_and_keepdims(self):
        value = ts.Tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            shape=(2, 2, 2),
        )

        result = ts.logsumexp(value, axis=[0, 2], keepdims=True)

        self.assertEqual(result.shape, (1, 2, 1))
        self.assertAlmostEqual(
            result.tolist()[0],
            math.log(sum(math.exp(item) for item in [0.0, 1.0, 4.0, 5.0])),
        )
        self.assertAlmostEqual(
            result.tolist()[1],
            math.log(sum(math.exp(item) for item in [2.0, 3.0, 6.0, 7.0])),
        )

    def test_logsumexp_gradient_is_softmax_and_supports_higher_derivatives(self):
        value = ts.Variable([[0.2, -0.4, 0.7]])
        output = ts.logsumexp(value, axis=1)

        first = ts.grad(output, value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([[1.0, 1.0, 1.0]]),
        )
        expected = ts.softmax(value.data, axis=1).tolist()

        self.assertValuesAlmostEqual(first.data.tolist(), expected)
        self.assertValuesAlmostEqual(second.tolist(), [0.0, 0.0, 0.0])

    def test_logsumexp_create_graph_handles_positive_infinity(self):
        value = ts.Variable([math.inf, 1.0])
        output = ts.logsumexp(value)

        first = ts.grad(output, value, create_graph=True)

        self.assertValuesAlmostEqual(first.data.tolist(), [1.0, 0.0])

    def test_log_softmax_is_stable_and_normalized(self):
        result = ts.log_softmax(
            ts.Tensor([[1000.0, 1001.0], [-1001.0, -1000.0]]),
            axis=1,
        )

        self.assertTrue(all(math.isfinite(item) for item in result.tolist()))
        for row in (result.tolist()[:2], result.tolist()[2:]):
            self.assertAlmostEqual(sum(math.exp(item) for item in row), 1.0)

    def test_log_softmax_gradient_matches_gradcheck(self):
        value = ts.Tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])

        self.assertTrue(ts.gradcheck(
            lambda x: ts.log_softmax(x, axis=1) ** 2.0,
            value,
        ))

    def test_cross_entropy_accepts_class_indices(self):
        logits = ts.Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

        losses = ts.cross_entropy(
            logits,
            ts.Tensor([0, 2], dtype=ts.int64),
            reduction="none",
        )
        expected = math.log(1.0 + math.exp(-1.0) + math.exp(-2.0))

        self.assertEqual(losses.shape, (2,))
        self.assertValuesAlmostEqual(losses.tolist(), [expected, expected])

    def test_cross_entropy_supports_a_nonfinal_class_axis(self):
        logits = ts.Tensor(
            [
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
                [[1.0, 3.0, 0.0], [2.0, 1.0, 4.0]],
            ]
        )
        targets = ts.Tensor([[0, 1, 0], [1, 0, 1]], dtype=ts.int64)

        result = ts.cross_entropy(
            logits, targets, axis=1, reduction="none"
        )

        self.assertEqual(result.shape, (2, 3))
        self.assertTrue(all(item >= 0.0 for item in result.tolist()))

    def test_cross_entropy_is_stable_for_extreme_logits(self):
        logits = ts.Variable([[1000.0, -1000.0], [-1000.0, 1000.0]])
        targets = ts.Tensor([0, 1], dtype=ts.int64)

        loss = ts.cross_entropy(logits, targets)
        gradient = ts.grad(loss, logits)

        self.assertTrue(math.isfinite(loss.data.item()))
        self.assertAlmostEqual(loss.data.item(), 0.0)
        self.assertTrue(all(math.isfinite(item) for item in gradient.tolist()))

    def test_cross_entropy_avoids_zero_times_infinity_at_float_limit(self):
        logits = ts.Variable([[1e308, -1e308]])

        loss = ts.cross_entropy(logits, [0])
        gradient = ts.grad(loss, logits)

        self.assertEqual(loss.data.item(), 0.0)
        self.assertValuesAlmostEqual(gradient.tolist(), [0.0, 0.0])

    def test_softmax_and_log_softmax_handle_positive_infinities(self):
        logits = ts.Tensor([[math.inf, 1.0, math.inf]])

        probabilities = ts.softmax(logits, axis=1)
        log_probabilities = ts.log_softmax(logits, axis=1)

        self.assertValuesAlmostEqual(probabilities.tolist(), [0.5, 0.0, 0.5])
        self.assertValuesAlmostEqual(
            [log_probabilities.tolist()[0], log_probabilities.tolist()[2]],
            [-math.log(2.0), -math.log(2.0)],
        )
        self.assertEqual(log_probabilities.tolist()[1], -math.inf)

    def test_probability_normalizers_propagate_nan_with_infinity(self):
        values = ts.Variable([
            [math.inf, math.nan],
            [math.nan, math.inf],
        ])

        probabilities = ts.softmax(values, axis=1)
        log_probabilities = ts.log_softmax(values, axis=1)
        normalizers = ts.logsumexp(values, axis=1)
        normalizer_gradient = ts.grad(
            normalizers,
            values,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertTrue(
            all(math.isnan(item) for item in probabilities.data._data)
        )
        self.assertTrue(
            all(math.isnan(item) for item in log_probabilities.data._data)
        )
        self.assertTrue(
            all(math.isnan(item) for item in normalizers.data._data)
        )
        self.assertTrue(
            all(math.isnan(item) for item in normalizer_gradient._data)
        )

    def test_probability_normalizers_reject_an_all_negative_infinity_axis(self):
        logits = ts.Tensor([[-math.inf, -math.inf]])

        with self.assertRaisesRegex(ValueError, "every value"):
            ts.softmax(logits, axis=1)
        with self.assertRaisesRegex(ValueError, "every value"):
            ts.log_softmax(logits, axis=1)

    def test_cross_entropy_dense_targets_are_differentiable(self):
        logits = ts.Variable([[0.2, -0.4, 0.7]])
        targets = ts.Tensor([[0.2, 0.3, 0.5]])

        self.assertTrue(ts.gradcheck(
            lambda x: ts.cross_entropy(x, targets),
            logits.data,
        ))

    def test_cross_entropy_supports_higher_derivatives(self):
        logits = ts.Variable([[0.0, 0.0]])
        loss = ts.cross_entropy(logits, [0])

        first = ts.grad(loss, logits, create_graph=True)
        second = ts.grad(first[0, 0], logits)

        self.assertValuesAlmostEqual(first.data.tolist(), [-0.5, 0.5])
        self.assertValuesAlmostEqual(second.tolist(), [0.25, -0.25])

    def test_cross_entropy_accepts_plain_nested_lists(self):
        class_index_loss = ts.cross_entropy(
            [[2.0, 1.0, 0.0]], [0], reduction="sum"
        )
        dense_loss = ts.cross_entropy(
            [[2.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]], reduction="sum"
        )

        self.assertAlmostEqual(class_index_loss.item(), dense_loss.item())

    def test_cross_entropy_distinguishes_broadcast_dense_targets(self):
        logits = ts.Tensor([[2.0, 0.0], [0.0, 2.0]])
        shared_targets = ts.Tensor([0.25, 0.75])
        expanded_targets = ts.Tensor([[0.25, 0.75], [0.25, 0.75]])

        shared_loss = ts.cross_entropy(
            logits,
            shared_targets,
            reduction="none",
        )
        expanded_loss = ts.cross_entropy(
            logits,
            expanded_targets,
            reduction="none",
        )

        self.assertValuesAlmostEqual(
            shared_loss.tolist(),
            expanded_loss.tolist(),
        )

    def test_binary_cross_entropy_from_logits_is_stable(self):
        logits = ts.Variable([1000.0, -1000.0, 1000.0, -1000.0])
        targets = ts.Tensor([1.0, 0.0, 0.0, 1.0])

        losses = ts.binary_cross_entropy(
            logits, targets, from_logits=True, reduction="none"
        )
        gradient = ts.grad(ts.sum(losses), logits)

        self.assertValuesAlmostEqual(losses.data.tolist()[:2], [0.0, 0.0])
        self.assertValuesAlmostEqual(losses.data.tolist()[2:], [1000.0, 1000.0])
        self.assertValuesAlmostEqual(gradient.tolist(), [0.0, 0.0, 1.0, -1.0])

    def test_loss_means_avoid_intermediate_overflow(self):
        logits = ts.Variable([[0.0, -1.0e308], [0.0, -1.0e308]])
        binary_logits = ts.Variable([1.0e308, 1.0e308])

        multiclass_loss = ts.cross_entropy(logits, [1, 1])
        binary_loss = ts.binary_cross_entropy(
            binary_logits,
            [0.0, 0.0],
            from_logits=True,
        )

        self.assertEqual(multiclass_loss.data.item(), 1.0e308)
        self.assertEqual(binary_loss.data.item(), 1.0e308)
        self.assertValuesAlmostEqual(
            ts.grad(multiclass_loss, logits).tolist(),
            [0.5, -0.5, 0.5, -0.5],
        )
        self.assertValuesAlmostEqual(
            ts.grad(binary_loss, binary_logits).tolist(),
            [0.5, 0.5],
        )

    def test_loss_sums_return_infinity_instead_of_raising(self):
        multiclass_loss = ts.cross_entropy(
            [[0.0, -1.0e308], [0.0, -1.0e308]],
            [1, 1],
            reduction="sum",
        )
        binary_loss = ts.binary_cross_entropy(
            [1.0e308, 1.0e308],
            [0.0, 0.0],
            from_logits=True,
            reduction="sum",
        )

        self.assertEqual(multiclass_loss.item(), math.inf)
        self.assertEqual(binary_loss.item(), math.inf)

    def test_binary_cross_entropy_from_logits_handles_exact_infinities(self):
        logits = ts.Variable([math.inf, -math.inf])
        targets = ts.Tensor([1.0, 0.0])

        loss = ts.binary_cross_entropy(
            logits,
            targets,
            from_logits=True,
            reduction="sum",
        )
        gradient = ts.grad(loss, logits)

        self.assertEqual(loss.data.item(), 0.0)
        self.assertValuesAlmostEqual(gradient.tolist(), [0.0, 0.0])

    def test_binary_cross_entropy_accepts_exact_boundary_probabilities(self):
        probabilities = ts.Variable([0.0, 1.0])
        targets = ts.Tensor([0.0, 1.0])

        loss = ts.binary_cross_entropy(
            probabilities, targets, reduction="sum"
        )
        gradient = ts.grad(loss, probabilities)

        self.assertEqual(loss.data.item(), 0.0)
        self.assertValuesAlmostEqual(gradient.tolist(), [1.0, -1.0])

    def test_binary_cross_entropy_supports_broadcasting_and_reductions(self):
        probabilities = ts.Variable([[0.8], [0.25]])
        targets = ts.Tensor([[1.0, 0.0]])

        none = ts.binary_cross_entropy(
            probabilities, targets, reduction="none"
        )
        total = ts.binary_cross_entropy(
            probabilities, targets, reduction="sum"
        )

        self.assertEqual(none.shape, (2, 2))
        self.assertAlmostEqual(total.data.item(), sum(none.data.tolist()))
        self.assertEqual(ts.grad(total, probabilities).shape, probabilities.shape)

    def test_binary_cross_entropy_from_logits_supports_higher_derivatives(self):
        logits = ts.Variable([0.0])
        loss = ts.binary_cross_entropy(
            logits, ts.Tensor([1.0]), from_logits=True
        )

        first = ts.grad(loss, logits, create_graph=True)
        second = ts.grad(first, logits)

        self.assertAlmostEqual(first.data.item(), -0.5)
        self.assertAlmostEqual(second.item(), 0.25)

    def test_binary_cross_entropy_boundary_gradients_are_differentiable(self):
        probabilities = ts.Variable([0.0, 1.0])
        loss = ts.binary_cross_entropy(
            probabilities,
            ts.Tensor([0.0, 1.0]),
            reduction="sum",
        )

        first = ts.grad(loss, probabilities, create_graph=True)
        second = ts.grad(
            first,
            probabilities,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertValuesAlmostEqual(first.data.tolist(), [1.0, -1.0])
        self.assertValuesAlmostEqual(second.tolist(), [1.0, 1.0])

    def test_binary_cross_entropy_broadcast_target_gradient_is_differentiable(self):
        logits = ts.Variable([[0.2], [-0.4]])
        targets = ts.Variable([[0.0, 1.0, 0.5]])
        loss = ts.binary_cross_entropy(
            logits, targets, from_logits=True, reduction="sum"
        )

        first = ts.grad(loss, targets, create_graph=True)
        second = ts.grad(
            first,
            logits,
            grad_outputs=ts.Tensor([[1.0, 1.0, 1.0]]),
        )

        self.assertEqual(first.shape, targets.shape)
        self.assertEqual(second.shape, logits.shape)
        self.assertValuesAlmostEqual(second.tolist(), [-3.0, -3.0])

    def test_losses_validate_targets_and_reduction(self):
        with self.assertRaisesRegex(ValueError, "Class index"):
            ts.cross_entropy([[1.0, 2.0]], [2])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            ts.binary_cross_entropy([0.5], [1.5])
        with self.assertRaisesRegex(ValueError, "reduction"):
            ts.cross_entropy([[1.0, 2.0]], [0], reduction="median")
        with self.assertRaisesRegex(TypeError, "from_logits"):
            ts.binary_cross_entropy([0.0], [1.0], from_logits=1)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            ts.cross_entropy([[1.0, 2.0]], [[0.2, 0.2]])


if __name__ == "__main__":
    unittest.main()
