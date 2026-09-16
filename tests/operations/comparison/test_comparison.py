import math
import unittest

import tensors as ts


class ComparisonTests(unittest.TestCase):
    def test_comparisons_broadcast_and_return_uint8_masks(self):
        left = ts.Tensor([[1.0], [3.0]])
        right = ts.Tensor([2.0, 3.0])

        result = ts.less(left, right)

        self.assertEqual(result.shape, (2, 2))
        self.assertIs(result.dtype, ts.uint8)
        self.assertEqual(result.tolist(), [1, 1, 0, 0])

    def test_all_comparison_functions_have_expected_values(self):
        left = [1.0, 2.0, 3.0]
        right = [2.0, 2.0, 2.0]

        self.assertEqual(ts.equal(left, right).tolist(), [0, 1, 0])
        self.assertEqual(ts.not_equal(left, right).tolist(), [1, 0, 1])
        self.assertEqual(ts.less(left, right).tolist(), [1, 0, 0])
        self.assertEqual(ts.less_equal(left, right).tolist(), [1, 1, 0])
        self.assertEqual(ts.greater(left, right).tolist(), [0, 0, 1])
        self.assertEqual(ts.greater_equal(left, right).tolist(), [0, 1, 1])

    def test_comparing_variables_produces_a_nondifferentiable_tensor(self):
        left = ts.Variable([1.0, 3.0])
        right = ts.Variable([2.0, 2.0])

        result = ts.greater(left, right)

        self.assertIsInstance(result, ts.Tensor)
        self.assertNotIsInstance(result, ts.Variable)
        self.assertEqual(result.tolist(), [0, 1])

    def test_named_comparisons_do_not_change_tensor_equality_semantics(self):
        left = ts.Tensor([1.0, 2.0])
        right = ts.Tensor([1.0, 2.0])

        self.assertTrue(left == right)
        self.assertEqual(ts.equal(left, right).tolist(), [1, 1])

    def test_nan_comparisons_follow_python_numeric_semantics(self):
        nan = math.nan

        self.assertEqual(ts.equal([nan], [nan]).tolist(), [0])
        self.assertEqual(ts.not_equal([nan], [nan]).tolist(), [1])
        self.assertEqual(ts.less([nan], [1.0]).tolist(), [0])

    def test_comparison_preserves_large_integer_scalar_exactly(self):
        value = 2 ** 60 + 1
        tensor = ts.Tensor([value], dtype=ts.int64)

        self.assertEqual(ts.equal(tensor, value).tolist(), [1])


if __name__ == "__main__":
    unittest.main()
