import unittest

import tensors as ts


class LinalgTests(unittest.TestCase):
    def test_linalg_namespace_exposes_dot(self):
        left = ts.Tensor([[1.0, 2.0]])
        right = ts.Tensor([[3.0], [4.0]])

        self.assertEqual(ts.linalg.dot(left, right).tolist(), [11.0])

    def test_dot_preserves_promoted_dtype(self):
        left = ts.Tensor([[1.0, 2.0]], dtype=ts.float32)
        right = ts.Tensor([[3.0], [4.0]], dtype=ts.float64)

        result = ts.linalg.dot(left, right)

        self.assertIs(result.dtype, ts.float64)

    def test_dot_rejects_non_2d_inputs(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            ts.linalg.dot(ts.Tensor([1.0, 2.0]), ts.Tensor([3.0, 4.0]))

    def test_dot_rejects_inner_dimension_mismatch(self):
        left = ts.Tensor([[1.0, 2.0, 3.0]])
        right = ts.Tensor([[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            ts.linalg.dot(left, right)


if __name__ == "__main__":
    unittest.main()
