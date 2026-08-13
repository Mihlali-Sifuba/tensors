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

    def test_min_and_max_are_axis_aware(self):
        matrix = ts.Tensor([[3.0, 1.0, 2.0], [4.0, 6.0, 5.0]])

        minimum = ts.min(matrix, axis=1, keepdims=True)
        maximum = ts.max(matrix, axis=0)

        self.assertEqual(minimum.shape, (2, 1))
        self.assertEqual(minimum.tolist(), [1.0, 4.0])
        self.assertEqual(maximum.shape, (3,))
        self.assertEqual(maximum.tolist(), [4.0, 6.0, 5.0])

    def test_axis_extrema_are_differentiable(self):
        minimum_input = ts.Variable([[1.0, 1.0], [2.0, 3.0]])
        maximum_input = ts.Variable([[1.0, 4.0], [3.0, 4.0]])

        ts.backward(ts.sum(ts.min(minimum_input, axis=1)))
        ts.backward(ts.sum(ts.max(maximum_input, axis=0)))

        self.assertEqual(minimum_input.grad.tolist(), [0.5, 0.5, 1.0, 0.0])
        self.assertEqual(maximum_input.grad.tolist(), [0.0, 0.5, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
