"""Absolute value and its subgradient at zero."""

import math
import unittest

import tensors as ts


class AbsoluteValueTests(unittest.TestCase):
    def test_abs_preserves_shape_and_dtype(self):
        value = ts.Tensor([[-2.0, 0.0], [3.0, -4.0]], dtype=ts.float32)

        result = abs(value)

        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [2.0, 0.0, 3.0, 4.0])

    def test_abs_uses_zero_subgradient_at_zero(self):
        value = ts.Variable([-2.0, 0.0, 3.0])

        first = ts.grad(ts.sum(ts.abs(value)), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(second.tolist(), [0.0, 0.0, 0.0])

    def test_abs_propagates_nan_to_value_and_gradient(self):
        value = ts.Variable([math.nan])
        result = ts.abs(value)

        ts.backward(result)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(value.grad.item()))

    def test_abs_rejects_higher_derivative_at_nan(self):
        value = ts.Variable([math.nan])

        with self.assertRaisesRegex(ValueError, "undefined at NaN"):
            ts.grad(ts.abs(value), value, create_graph=True)


if __name__ == "__main__":
    unittest.main()
