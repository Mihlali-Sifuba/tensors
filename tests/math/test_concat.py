import unittest

import tensors as ts


class ConcatTests(unittest.TestCase):
    def test_concat_along_first_axis(self):
        result = ts.concat([ts.Tensor([[1, 2]]), ts.Tensor([[3, 4]])])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_concat_along_nonzero_axis_uses_each_tensor_block_size(self):
        left = ts.Tensor([[1, 2], [3, 4]])
        right = ts.Tensor([[10, 20, 30], [40, 50, 60]])

        result = ts.concat([left, right], axis=1)

        self.assertEqual(result.shape, (2, 5))
        self.assertEqual(
            result.tolist(),
            [1.0, 2.0, 10.0, 20.0, 30.0, 3.0, 4.0, 40.0, 50.0, 60.0],
        )

    def test_concat_accepts_negative_axis(self):
        result = ts.concat([ts.Tensor([[1, 2]]), ts.Tensor([[3, 4]])], axis=-1)

        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_concat_promotes_mixed_input_dtypes(self):
        left = ts.Tensor([1], dtype=ts.int32)
        right = ts.Tensor([2.5], dtype=ts.float64)

        result = ts.concat([left, right])

        self.assertIs(result.dtype, ts.float64)
        self.assertEqual(result.tolist(), [1.0, 2.5])

        reverse = ts.concat([right, left])

        self.assertIs(reverse.dtype, ts.float64)
        self.assertEqual(reverse.tolist(), [2.5, 1.0])

    def test_concat_validates_input_shapes(self):
        with self.assertRaisesRegex(ValueError, "non-concat"):
            ts.concat([ts.Tensor([[1, 2]]), ts.Tensor([[3], [4]])], axis=1)

    def test_concat_rejects_non_integer_axes(self):
        tensors = [ts.Tensor([1]), ts.Tensor([2])]

        with self.assertRaisesRegex(TypeError, "integer"):
            ts.concat(tensors, axis=False)
        with self.assertRaisesRegex(TypeError, "integer"):
            ts.concat(tensors, axis=0.0)

    def test_math_namespace_exposes_concat_operation_class(self):
        result = ts.math.Concat.forward(ts.Tensor([1]), ts.Tensor([2]))

        self.assertEqual(result.tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
