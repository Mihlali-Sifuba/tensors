import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend

from ._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyLossTests(NumPyParityTestCase):
    """Normalization and cross-entropy kernels and their guards."""

    def test_fused_probability_loss_rejects_nonfinite_probabilities(self):
        probabilities = ts.Tensor([float("nan")] * 64)
        targets = ts.full((64,), 0.5)

        with ts.use_backend("numpy"):
            with self.assertRaisesRegex(
                ValueError,
                "probabilities must be between 0 and 1",
            ):
                ts.binary_cross_entropy(probabilities, targets)
    def test_normalization_and_loss_operations_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend,
                "normalization",
                wraps=numpy_backend.normalization,
            ) as normalization,
            patch.object(
                numpy_backend,
                "normalization_gradient",
                wraps=numpy_backend.normalization_gradient,
            ) as normalization_gradient,
            patch.object(
                numpy_backend,
                "logsumexp",
                wraps=numpy_backend.logsumexp,
            ) as logsumexp,
            patch.object(
                numpy_backend,
                "logsumexp_gradient",
                wraps=numpy_backend.logsumexp_gradient,
            ) as logsumexp_gradient,
            patch.object(
                numpy_backend,
                "cross_entropy",
                wraps=numpy_backend.cross_entropy,
            ) as cross_entropy,
            patch.object(
                numpy_backend,
                "cross_entropy_gradient",
                wraps=numpy_backend.cross_entropy_gradient,
            ) as cross_entropy_gradient,
            patch.object(
                numpy_backend,
                "one_hot_targets",
                wraps=numpy_backend.one_hot_targets,
            ) as one_hot_targets,
            patch.object(
                numpy_backend,
                "distributions_valid",
                wraps=numpy_backend.distributions_valid,
            ) as distributions_valid,
            patch.object(
                numpy_backend,
                "binary_cross_entropy",
                wraps=numpy_backend.binary_cross_entropy,
            ) as binary_cross_entropy,
            patch.object(
                numpy_backend,
                "binary_cross_entropy_gradient",
                wraps=numpy_backend.binary_cross_entropy_gradient,
            ) as binary_cross_entropy_gradient,
        ):
            with ts.use_backend("numpy"):
                logits = ts.Variable(ts.full((16, 4), 0.25))
                ts.grad(ts.sum(ts.softmax(logits, axis=1)), logits)
                ts.grad(ts.sum(ts.log_softmax(logits, axis=1)), logits)
                ts.grad(ts.sum(ts.logsumexp(logits, axis=1)), logits)

                classes = ts.Tensor([0] * 16, dtype=ts.int64)
                ts.grad(ts.cross_entropy(logits, classes), logits)

                binary_logits = ts.Variable(ts.full((64,), 0.25))
                binary_targets = ts.full((64,), 0.5)
                ts.grad(
                    ts.binary_cross_entropy(
                        binary_logits,
                        binary_targets,
                        from_logits=True,
                    ),
                    binary_logits,
                )

        self.assertEqual(normalization.call_count, 2)
        self.assertEqual(normalization_gradient.call_count, 2)
        logsumexp.assert_called_once()
        logsumexp_gradient.assert_called_once()
        cross_entropy.assert_called_once()
        cross_entropy_gradient.assert_called_once()
        one_hot_targets.assert_called_once()
        self.assertEqual(distributions_valid.call_count, 2)
        binary_cross_entropy.assert_called_once()
        binary_cross_entropy_gradient.assert_called_once()
    def test_normalization_and_losses_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                logits = ts.Variable(
                    ts.Tensor(
                        [float(index % 4) / 4.0 for index in range(64)],
                        shape=(16, 4),
                    )
                )
                softmax = ts.softmax(logits, axis=1)
                log_softmax = ts.log_softmax(logits, axis=1)
                normalizer = ts.logsumexp(logits, axis=1)
                classes = ts.Tensor([index % 4 for index in range(16)])
                multiclass = ts.cross_entropy(logits, classes)
                multiclass_gradient = ts.grad(multiclass, logits)

                predictions = ts.Variable(
                    ts.Tensor(
                        [0.2 + (index % 5) / 10.0 for index in range(64)]
                    )
                )
                targets = ts.full((64,), 0.5)
                binary = ts.binary_cross_entropy(predictions, targets)
                binary_gradient = ts.grad(binary, predictions)
                return (
                    softmax.data,
                    log_softmax.data,
                    normalizer.data,
                    multiclass.data,
                    multiclass_gradient,
                    binary.data,
                    binary_gradient,
                )

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)
    def test_dense_target_validation_dispatches_to_numpy(self):
        targets = (
            ts.full((16, 4), 0.25),
            ts.full((16, 4), 0.25),
        )
        targets[0][0, 0] = 0.5
        targets[1][0, 0] = 0.25000015
        with patch.object(
            numpy_backend,
            "distributions_valid",
            wraps=numpy_backend.distributions_valid,
        ) as distributions_valid:
            with ts.use_backend("numpy"):
                for target in targets:
                    with self.assertRaisesRegex(ValueError, "sum to 1"):
                        ts.cross_entropy(ts.full((16, 4), 1.0), target)

        self.assertEqual(distributions_valid.call_count, 2)


if __name__ == "__main__":
    unittest.main()
