"""Axis permutation and reinterpretation of a tensor's shape."""

import unittest

import tensors as ts


class TransposeAndReshapeTests(unittest.TestCase):

    def test_transpose_reports_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            ts.transpose(ts.Tensor([1, 2]))

    def test_transpose_swaps_2d_axes_and_preserves_dtype(self):
        matrix = ts.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ts.float32)

        result = ts.transpose(matrix)

        self.assertEqual(result.shape, (3, 2))
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_transpose_rejects_boolean_axes(self):
        with self.assertRaisesRegex(TypeError, "only integers"):
            ts.transpose(
                ts.Tensor([[1.0, 2.0], [3.0, 4.0]]),
                axes=(True, False),
            )

    def test_reshape_error_on_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Cannot reshape"):
            ts.reshape(ts.Tensor([1, 2, 3]), (2,))

    def test_reshape_preserves_dtype(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        result = ts.reshape(tensor, (2, 2))

        self.assertEqual(result.shape, (2, 2))
        self.assertIs(result.dtype, ts.float32)

    def test_math_namespace_exposes_reshape_operation_class(self):
        result = ts.math.Reshape(shape=(2, 2)).forward(ts.Tensor([1, 2, 3, 4]))

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
