"""The sign indicator and its subgradient."""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class SignTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_sign_returns_negative_zero_and_positive_indicators(self):
        value = ts.Tensor([-3, 0, 4], dtype=ts.int32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1, 0, 1])
        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.int32)

    def test_sign_preserves_float_dtype(self):
        value = ts.Tensor([-3.0, 0.0, 4.0], dtype=ts.float32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1.0, 0.0, 1.0])
        self.assertIs(result.dtype, ts.float32)

    def test_sign_propagates_nan_to_value_and_first_gradient(self):
        value = ts.Variable([math.nan])

        result = ts.sign(value)
        gradient = ts.grad(result, value)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(gradient.item()))

    def test_sign_derivative_is_zero_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        gradient = ts.grad(ts.sign(value), value)

        self.assertEqual(gradient.tolist(), [0.0, 0.0])

    def test_sign_derivative_rejects_zero(self):
        value = ts.Variable([0.0])

        with self.assertRaisesRegex(ValueError, "undefined at zero"):
            ts.grad(ts.sign(value), value)

    def test_sign_has_zero_higher_derivatives_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        first = ts.grad(ts.sign(value), value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [0.0, 0.0])
        self.assertEqual(second.tolist(), [0.0, 0.0])

    def test_sign_passes_gradcheck_away_from_zero(self):
        self.assertTrue(ts.gradcheck(ts.sign, ts.Tensor([-2.0, 3.0])))

    def test_public_math_namespace_exposes_sign_function_and_class(self):
        self.assertEqual(ts.math.sign([-1.0, 0.0, 1.0]).tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(
            ts.math.Sign().forward(ts.Tensor([-1.0, 0.0, 1.0])).tolist(),
            [-1.0, 0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
