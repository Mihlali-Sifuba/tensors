import unittest

import tensors as ts


class ExtremaTests(unittest.TestCase):
    def test_min_returns_scalar_tensor_with_input_dtype(self):
        tensor = ts.Tensor([3, 1, 2], dtype=ts.int32)

        result = ts.min(tensor)

        self.assertEqual(result.shape, (1,))
        self.assertIs(result.dtype, ts.int32)
        self.assertEqual(result.tolist(), [1])

    def test_max_returns_scalar_tensor_with_input_dtype(self):
        tensor = ts.Tensor([3.0, 1.0, 2.0], dtype=ts.float32)

        result = ts.max(tensor)

        self.assertEqual(result.shape, (1,))
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [3.0])

    def test_min_and_max_accept_plain_lists(self):
        self.assertEqual(ts.min([3, 1, 2]).tolist(), [1.0])
        self.assertEqual(ts.max([3, 1, 2]).tolist(), [3.0])

    def test_math_namespace_exposes_extrema_operation_classes(self):
        self.assertEqual(ts.math.Min.forward(ts.Tensor([3, 1, 2])).tolist(), [1.0])
        self.assertEqual(ts.math.Max.forward(ts.Tensor([3, 1, 2])).tolist(), [3.0])

    def test_min_rejects_empty_tensor(self):
        with self.assertRaisesRegex(ValueError, "empty tensor"):
            ts.min(ts.Tensor([]))

    def test_max_rejects_empty_tensor(self):
        with self.assertRaisesRegex(ValueError, "empty tensor"):
            ts.max(ts.Tensor([]))

    def test_variable_extrema_are_not_differentiable_yet(self):
        variable = ts.Variable([1.0, 2.0])

        with self.assertRaises(NotImplementedError):
            ts.min(variable)
        with self.assertRaises(NotImplementedError):
            ts.max(variable)


if __name__ == "__main__":
    unittest.main()
