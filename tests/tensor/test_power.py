import unittest

import tensors as ts


class TensorPowerTests(unittest.TestCase):
    def test_scalar_exponent_preserves_integer_dtype_for_non_negative_powers(self):
        base = ts.Tensor([2, 3], dtype=ts.int32)

        result = base ** 3

        self.assertIs(result.dtype, ts.int32)
        self.assertEqual(result.tolist(), [8, 27])

    def test_tensor_exponent_uses_elementwise_broadcasting(self):
        base = ts.Tensor([[1.0], [2.0]])
        exponent = ts.Tensor([2.0, 3.0])

        result = base ** exponent

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 1.0, 4.0, 8.0])

    def test_reverse_power_uses_tensor_as_the_exponent(self):
        exponent = ts.Tensor([1.0, 2.0, 3.0])

        result = 2.0 ** exponent

        self.assertEqual(result.tolist(), [2.0, 4.0, 8.0])

    def test_package_power_function_delegates_to_operator(self):
        result = ts.pow(ts.Tensor([2.0, 3.0]), 2.0)

        self.assertEqual(result.tolist(), [4.0, 9.0])

    def test_fractional_power_of_negative_values_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "real-valued"):
            _ = ts.Tensor([-1.0]) ** 0.5


if __name__ == "__main__":
    unittest.main()
