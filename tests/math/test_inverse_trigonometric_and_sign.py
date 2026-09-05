import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class InverseTrigonometricTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_functions_return_elementwise_values_and_preserve_shape(self):
        values = ts.Tensor([[-0.5, 0.0], [0.5, 1.0]])

        inverse_sine = ts.arcsin(values)
        inverse_cosine = ts.arccos(values)
        inverse_tangent = ts.arctan(values)

        for result in (inverse_sine, inverse_cosine, inverse_tangent):
            self.assertEqual(result.shape, values.shape)
        for actual, expected in zip(inverse_sine.tolist(), map(math.asin, values.tolist())):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(inverse_cosine.tolist(), map(math.acos, values.tolist())):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(inverse_tangent.tolist(), map(math.atan, values.tolist())):
            self.assertAlmostEqual(actual, expected)

    def test_public_math_namespace_exposes_functions_and_classes(self):
        self.assertAlmostEqual(ts.math.arcsin([0.5]).item(), math.asin(0.5))
        self.assertAlmostEqual(ts.math.arccos([0.5]).item(), math.acos(0.5))
        self.assertAlmostEqual(ts.math.arctan([0.5]).item(), math.atan(0.5))
        self.assertAlmostEqual(
            ts.math.ArcSin().forward(ts.Tensor([0.5])).item(),
            math.asin(0.5),
        )
        self.assertAlmostEqual(
            ts.math.ArcCos().forward(ts.Tensor([0.5])).item(),
            math.acos(0.5),
        )
        self.assertAlmostEqual(
            ts.math.ArcTan().forward(ts.Tensor([0.5])).item(),
            math.atan(0.5),
        )

    def test_integer_inputs_promote_and_float_inputs_preserve_dtype(self):
        for function in (ts.arcsin, ts.arccos, ts.arctan):
            self.assertIs(function(ts.Tensor([0], dtype=ts.int32)).dtype, ts.float64)
            self.assertIs(function(ts.Tensor([0], dtype=ts.float32)).dtype, ts.float32)

    def test_arcsin_and_arccos_enforce_their_real_domain(self):
        for function in (ts.arcsin, ts.arccos):
            with self.subTest(function=function.__name__, value=-1.1):
                with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                    function([-1.1])
            with self.subTest(function=function.__name__, value=1.1):
                with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                    function([1.1])

    def test_arcsin_and_arccos_values_exist_at_domain_endpoints(self):
        self.assertEqual(ts.arcsin([-1.0, 1.0]).tolist(), [-math.pi / 2.0, math.pi / 2.0])
        self.assertEqual(ts.arccos([-1.0, 1.0]).tolist(), [math.pi, 0.0])

    def test_arcsin_and_arccos_derivatives_reject_domain_endpoints(self):
        for function in (ts.arcsin, ts.arccos):
            for point in (-1.0, 1.0):
                with self.subTest(function=function.__name__, point=point):
                    value = ts.Variable([point])
                    with self.assertRaisesRegex(ValueError, "undefined at -1 and 1"):
                        ts.grad(function(value), value)

    def test_first_derivatives(self):
        point = 0.4
        value = ts.Variable([point])

        arcsin_gradient = ts.grad(ts.arcsin(value), value)
        arccos_gradient = ts.grad(ts.arccos(value), value)
        arctan_gradient = ts.grad(ts.arctan(value), value)

        denominator = math.sqrt(1.0 - point ** 2.0)
        self.assertAlmostEqual(arcsin_gradient.item(), 1.0 / denominator)
        self.assertAlmostEqual(arccos_gradient.item(), -1.0 / denominator)
        self.assertAlmostEqual(arctan_gradient.item(), 1.0 / (1.0 + point ** 2.0))

    def test_arctan_gradient_does_not_overflow_for_large_inputs(self):
        value = ts.Variable([1.0e308, math.inf, -math.inf])

        gradient = ts.grad(ts.arctan(value), value)

        self.assertEqual(gradient.tolist(), [0.0, 0.0, 0.0])

    def test_second_derivatives(self):
        point = 0.4
        value = ts.Variable([point])

        arcsin_first = ts.grad(ts.arcsin(value), value, create_graph=True)
        arccos_first = ts.grad(ts.arccos(value), value, create_graph=True)
        arctan_first = ts.grad(ts.arctan(value), value, create_graph=True)

        inverse_trig_denominator = (1.0 - point ** 2.0) ** 1.5
        self.assertAlmostEqual(
            ts.grad(arcsin_first, value).item(),
            point / inverse_trig_denominator,
        )
        self.assertAlmostEqual(
            ts.grad(arccos_first, value).item(),
            -point / inverse_trig_denominator,
        )
        self.assertAlmostEqual(
            ts.grad(arctan_first, value).item(),
            -2.0 * point / (1.0 + point ** 2.0) ** 2.0,
        )

    def test_computation_forward_replays_inverse_trigonometric_nodes(self):
        value = ts.Variable([0.0])
        output = ts.arcsin(value) + ts.arccos(value) + ts.arctan(value)
        computation = ts.graph.Computation(output)
        value.data = ts.Tensor([0.5])

        replayed = computation.forward()

        expected = math.asin(0.5) + math.acos(0.5) + math.atan(0.5)
        self.assertAlmostEqual(replayed.item(), expected)

    def test_composite_expression_passes_gradcheck(self):
        values = ts.Tensor([-0.7, 0.2, 0.8])

        self.assertTrue(
            ts.gradcheck(
                lambda value: (
                    ts.arcsin(value)
                    + ts.arccos(value)
                    + ts.arctan(value)
                ),
                values,
            )
        )


class SignTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_sign_returns_negative_zero_and_positive_indicators(self):
        value = ts.Tensor([-3, 0, 4], dtype=ts.int32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1, 0, 1])
        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.int32)

    def test_sign_preserves_float_dtype(self):
        value = ts.Tensor([-3.0, 0.0, 4.0], dtype=ts.float32)

        result = ts.sign(value)

        self.assertEqual(result.tolist(), [-1.0, 0.0, 1.0])
        self.assertIs(result.dtype, ts.float32)

    def test_sign_propagates_nan_to_value_and_first_gradient(self):
        value = ts.Variable([math.nan])

        result = ts.sign(value)
        gradient = ts.grad(result, value)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(gradient.item()))

    def test_sign_derivative_is_zero_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        gradient = ts.grad(ts.sign(value), value)

        self.assertEqual(gradient.tolist(), [0.0, 0.0])

    def test_sign_derivative_rejects_zero(self):
        value = ts.Variable([0.0])

        with self.assertRaisesRegex(ValueError, "undefined at zero"):
            ts.grad(ts.sign(value), value)

    def test_sign_has_zero_higher_derivatives_away_from_zero(self):
        value = ts.Variable([-2.0, 3.0])

        first = ts.grad(ts.sign(value), value, create_graph=True)
        second = ts.grad(
            first,
            value,
            grad_outputs=ts.Tensor([1.0, 1.0]),
        )

        self.assertEqual(first.data.tolist(), [0.0, 0.0])
        self.assertEqual(second.tolist(), [0.0, 0.0])

    def test_sign_passes_gradcheck_away_from_zero(self):
        self.assertTrue(ts.gradcheck(ts.sign, ts.Tensor([-2.0, 3.0])))

    def test_public_math_namespace_exposes_sign_function_and_class(self):
        self.assertEqual(ts.math.sign([-1.0, 0.0, 1.0]).tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(
            ts.math.Sign().forward(ts.Tensor([-1.0, 0.0, 1.0])).tolist(),
            [-1.0, 0.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
