import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VarianceTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_variance_accepts_plain_lists(self):
        result = ts.variance([1.0, 2.0, 3.0])

        self.assertAlmostEqual(result.item(), 2.0 / 3.0)
        self.assertEqual(result.shape, (1,))

    def test_variance_is_axis_aware_and_supports_keepdims(self):
        matrix = ts.Tensor([[1.0, 3.0], [2.0, 6.0]])

        result = ts.variance(matrix, axis=1, keepdims=True)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.tolist(), [1.0, 4.0])

    def test_variance_supports_multiple_axes(self):
        value = ts.Tensor(
            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
            shape=(2, 2, 2),
        )

        result = ts.variance(value, axis=(0, 2))

        self.assertEqual(result.shape, (2,))
        for item in result.tolist():
            self.assertAlmostEqual(item, 17.0)

    def test_integer_inputs_promote_and_float_inputs_preserve_dtype(self):
        integer_result = ts.variance(ts.Tensor([1, 2], dtype=ts.int32))
        float_result = ts.variance(ts.Tensor([1.0, 2.0], dtype=ts.float32))

        self.assertIs(integer_result.dtype, ts.float64)
        self.assertIs(float_result.dtype, ts.float32)

    def test_empty_variance_returns_nan_scalar(self):
        result = ts.variance(ts.Tensor([]))

        self.assertEqual(result.shape, (1,))
        self.assertTrue(math.isnan(result.item()))

    def test_variance_avoids_intermediate_overflow(self):
        value = ts.Variable([1.0e308, -1.0e308])

        result = ts.variance(value)
        gradient = ts.grad(result, value)

        self.assertEqual(result.data.item(), math.inf)
        self.assertEqual(gradient.tolist(), [1.0e308, -1.0e308])

    def test_first_derivative_matches_population_variance_formula(self):
        value = ts.Variable([1.0, 2.0, 3.0])

        gradient = ts.grad(ts.variance(value), value)

        self.assertAlmostEqual(gradient.tolist()[0], -2.0 / 3.0)
        self.assertAlmostEqual(gradient.tolist()[1], 0.0)
        self.assertAlmostEqual(gradient.tolist()[2], 2.0 / 3.0)

    def test_zero_variance_has_zero_gradient_and_valid_hessian(self):
        value = ts.Variable([4.0, 4.0])

        output = ts.variance(value)
        gradient = ts.grad(output, value)
        hessian = ts.hessian(output, value)

        self.assertEqual(output.data.tolist(), [0.0])
        self.assertEqual(gradient.tolist(), [0.0, 0.0])
        self.assertEqual(hessian.tolist(), [0.5, -0.5, -0.5, 0.5])

    def test_singleton_variance_has_zero_first_and_second_derivatives(self):
        value = ts.Variable([4.0])
        output = ts.variance(value)

        first = ts.grad(output, value, create_graph=True)
        second = ts.grad(first, value)

        self.assertEqual(first.data.tolist(), [0.0])
        self.assertEqual(second.tolist(), [0.0])

    def test_axis_variance_passes_gradcheck(self):
        value = ts.Tensor([[1.0, 2.0, 4.0], [2.0, 5.0, 9.0]])

        self.assertTrue(ts.gradcheck(lambda item: ts.variance(item, axis=1), value))

    def test_computation_forward_replays_variance(self):
        value = ts.Variable([1.0, 3.0])
        output = ts.variance(value)
        computation = ts.graph.Computation(output)
        value.data = ts.Tensor([2.0, 6.0])

        replayed = computation.forward()

        self.assertEqual(replayed.tolist(), [4.0])
        self.assertEqual(ts.grad(output, value).tolist(), [-2.0, 2.0])

    def test_math_namespace_exposes_variance_function_and_class(self):
        self.assertAlmostEqual(ts.math.variance([1.0, 2.0, 3.0]).item(), 2.0 / 3.0)
        self.assertAlmostEqual(
            ts.math.Variance().forward(ts.Tensor([1.0, 2.0, 3.0])).item(),
            2.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
