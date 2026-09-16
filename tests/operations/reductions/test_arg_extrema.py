"""Indices of extreme values along an axis."""

import math
import unittest

import tensors as ts


class ArgExtremaTests(unittest.TestCase):
    def test_argmin_and_argmax_return_first_ties_as_int64(self):
        value = ts.Tensor([3.0, 1.0, 1.0, 4.0, 4.0])

        minimum = ts.argmin(value)
        maximum = ts.argmax(value)

        self.assertIs(minimum.dtype, ts.int64)
        self.assertIs(maximum.dtype, ts.int64)
        self.assertEqual(minimum.tolist(), [1])
        self.assertEqual(maximum.tolist(), [3])

    def test_argmin_and_argmax_are_axis_aware(self):
        value = ts.Tensor([[3.0, 1.0], [0.0, 4.0]])

        self.assertEqual(ts.argmin(value, axis=1).tolist(), [1, 0])
        self.assertEqual(ts.argmax(value, axis=0).tolist(), [0, 1])
        kept = ts.argmax(value, axis=1, keepdims=True)
        self.assertEqual(kept.shape, (2, 1))
        self.assertEqual(kept.tolist(), [0, 1])

    def test_arg_extrema_of_variable_are_nondifferentiable_tensors(self):
        value = ts.Variable([2.0, 1.0])

        result = ts.argmin(value)

        self.assertIsInstance(result, ts.Tensor)
        self.assertNotIsInstance(result, ts.Variable)

    def test_arg_extrema_reject_empty_inputs_and_invalid_axes(self):
        with self.assertRaisesRegex(ValueError, "empty tensor"):
            ts.argmax(ts.Tensor([]))
        with self.assertRaisesRegex(TypeError, "integer or None"):
            ts.argmin([1.0], axis=(0,))

    def test_arg_extrema_select_first_nan(self):
        value = ts.Tensor([2.0, math.nan, math.nan, 1.0])

        self.assertEqual(ts.argmin(value).tolist(), [1])
        self.assertEqual(ts.argmax(value).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
