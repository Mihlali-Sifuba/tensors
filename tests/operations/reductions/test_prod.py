"""The product reduction and its gradient around zeros."""

import unittest

import tensors as ts


class ProductTests(unittest.TestCase):
    def test_prod_is_axis_aware_and_supports_keepdims(self):
        value = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = ts.prod(value, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [6.0, 120.0])

    def test_prod_of_empty_tensor_uses_multiplicative_identity(self):
        result = ts.prod(ts.Tensor([]))

        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.tolist(), [1.0])

    def test_prod_gradient_handles_no_zero_one_zero_and_multiple_zeros(self):
        cases = (
            ([2.0, 3.0, 4.0], [12.0, 8.0, 6.0]),
            ([2.0, 0.0, 4.0], [0.0, 8.0, 0.0]),
            ([0.0, 3.0, 0.0], [0.0, 0.0, 0.0]),
        )
        for values, expected in cases:
            with self.subTest(values=values):
                value = ts.Variable(values)
                ts.backward(ts.prod(value))
                self.assertEqual(value.grad.tolist(), expected)

    def test_prod_higher_derivative_contains_cross_terms(self):
        value = ts.Variable([2.0, 3.0])

        first = ts.grad(ts.prod(value), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [3.0, 2.0])
        self.assertEqual(second.tolist(), [1.0, 1.0])

    def test_prod_axis_gradient_is_group_local(self):
        value = ts.Variable([[2.0, 3.0], [4.0, 5.0]])

        ts.backward(ts.sum(ts.prod(value, axis=1)))

        self.assertEqual(value.grad.tolist(), [3.0, 2.0, 5.0, 4.0])


if __name__ == "__main__":
    unittest.main()
