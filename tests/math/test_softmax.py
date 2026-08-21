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
        values = ts.Tensor([
            [math.inf, math.nan],
            [math.nan, math.inf],
        ])

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


if __name__ == "__main__":
    unittest.main()
