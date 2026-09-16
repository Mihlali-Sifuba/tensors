import math
import unittest

import tensors as ts


class SoftmaxTests(unittest.TestCase):
    def test_softmax_normalizes_a_vector(self):
        result = ts.softmax(ts.Tensor([1.0, 2.0, 3.0]))

        normalizer = sum(math.exp(value) for value in [1.0, 2.0, 3.0])
        expected = [math.exp(value) / normalizer for value in [1.0, 2.0, 3.0]]
        self.assertAlmostEqual(sum(result.tolist()), 1.0)
        for actual, expected_value in zip(result.tolist(), expected):
            self.assertAlmostEqual(actual, expected_value)

    def test_softmax_uses_the_final_axis_by_default(self):
        result = ts.softmax(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))

        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result.tolist()[0] + result.tolist()[1], 1.0)
        self.assertAlmostEqual(result.tolist()[2] + result.tolist()[3], 1.0)

    def test_softmax_supports_an_explicit_axis(self):
        result = ts.softmax(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]), axis=0)

        self.assertAlmostEqual(result.tolist()[0] + result.tolist()[2], 1.0)
        self.assertAlmostEqual(result.tolist()[1] + result.tolist()[3], 1.0)

    def test_softmax_is_stable_for_large_values(self):
        result = ts.softmax(ts.Tensor([1000.0, 1001.0]))

        self.assertAlmostEqual(result.tolist()[0], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(result.tolist()[1], math.e / (1.0 + math.e))

    def test_softmax_propagates_nan_even_with_positive_infinity(self):
        values = ts.Tensor(
            [
                [math.inf, math.nan],
                [math.nan, math.inf],
            ]
        )

        result = ts.softmax(values, axis=1)

        self.assertTrue(all(math.isnan(item) for item in result._data))

    def test_softmax_validates_axis_and_empty_axis(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.softmax(ts.Tensor([1.0]), axis=1)
        with self.assertRaisesRegex(ValueError, "empty axis"):
            ts.softmax(ts.Tensor([]))
        with self.assertRaisesRegex(TypeError, "integer"):
            ts.softmax(ts.Tensor([1.0]), axis=False)
        with self.assertRaisesRegex(TypeError, "integer"):
            ts.softmax(ts.Tensor([1.0]), axis=0.0)


class IntegerInputNormalizationGradientTests(unittest.TestCase):
    """Softmax-family VJPs over an integer-valued input.

    The probabilities these VJPs weight by are fractions, so they have to be
    evaluated in floating point whatever the input dtype. Computing them at
    the input's own dtype truncates every probability to zero or one, which
    silently returns a zero gradient rather than raising.

    A Variable must be floating point to require gradients, so these enter
    through the dispatch entry point the operations use, which is the
    boundary the kernels are written against.
    """

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _reference(self, values, grad_values, axis, kernel):
        floating = ts.Tensor([float(value) for value in values])
        return kernel(ts.Tensor(grad_values), floating, axis)

    def test_softmax_gradient_does_not_truncate_integer_inputs(self):
        from tensors.backend import execute_softmax_gradient

        for dtype in (ts.int32, ts.int64):
            with self.subTest(dtype=dtype.name):
                value = ts.Tensor([1, 2, 3, 4], dtype=dtype)
                grad = ts.Tensor([1.0, 0.0, 0.0, 0.0])

                storage = execute_softmax_gradient(grad, value, 0)
                result = list(storage.buffer)

                self.assertTrue(
                    any(item != 0.0 for item in result),
                    "an integer input truncated the softmax probabilities",
                )
                expected = self._reference(
                    [1, 2, 3, 4], [1.0, 0.0, 0.0, 0.0], 0, execute_softmax_gradient
                )
                for actual, expected_value in zip(result, expected.buffer):
                    self.assertAlmostEqual(actual, expected_value)

    def test_log_softmax_gradient_does_not_truncate_integer_inputs(self):
        from tensors.backend import execute_log_softmax_gradient

        for dtype in (ts.int32, ts.int64):
            with self.subTest(dtype=dtype.name):
                value = ts.Tensor([1, 2, 3, 4], dtype=dtype)
                grad = ts.Tensor([1.0, 0.0, 0.0, 0.0])

                storage = execute_log_softmax_gradient(grad, value, 0)
                result = list(storage.buffer)

                # Every off-diagonal term is -g_i * p_j, which truncation
                # flattened to zero while leaving the first term intact.
                self.assertTrue(
                    all(item != 0.0 for item in result[1:]),
                    "an integer input truncated the log-softmax probabilities",
                )
                expected = self._reference(
                    [1, 2, 3, 4],
                    [1.0, 0.0, 0.0, 0.0],
                    0,
                    execute_log_softmax_gradient,
                )
                for actual, expected_value in zip(result, expected.buffer):
                    self.assertAlmostEqual(actual, expected_value)

    def test_integer_and_floating_inputs_agree_on_every_backend(self):
        from tensors.backend import (
            execute_log_softmax_gradient,
            execute_softmax_gradient,
        )

        values = [1, 2, 3, 4] * 16
        grad_values = [1.0, -2.0, 0.5, 0.25] * 16
        for backend in ts.available_backends():
            for kernel in (execute_softmax_gradient, execute_log_softmax_gradient):
                with self.subTest(backend=backend, kernel=kernel.__name__):
                    with ts.use_backend(backend):
                        integer = kernel(
                            ts.Tensor(grad_values),
                            ts.Tensor(values, dtype=ts.int64),
                            0,
                        )
                        floating = kernel(
                            ts.Tensor(grad_values),
                            ts.Tensor([float(value) for value in values]),
                            0,
                        )
                    for actual, expected in zip(integer.buffer, floating.buffer):
                        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
