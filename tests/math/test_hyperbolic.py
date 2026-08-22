import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class HyperbolicTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_functions_return_elementwise_values_and_preserve_shape(self):
        values = ts.Tensor([[-1.0, 0.0], [0.5, 1.0]])

        hyperbolic_sine = ts.sinh(values)
        hyperbolic_cosine = ts.cosh(values)
        inverse_hyperbolic_sine = ts.arcsinh(values)

        for result in (
            hyperbolic_sine,
            hyperbolic_cosine,
            inverse_hyperbolic_sine,
        ):
            self.assertEqual(result.shape, values.shape)
        for actual, expected in zip(
            hyperbolic_sine.tolist(),
            map(math.sinh, values.tolist()),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            hyperbolic_cosine.tolist(),
            map(math.cosh, values.tolist()),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            inverse_hyperbolic_sine.tolist(),
            map(math.asinh, values.tolist()),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_inverse_functions_return_values_on_their_domains(self):
        arccosh_values = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        arctanh_values = ts.Tensor([[-0.75, 0.0], [0.25, 0.75]])

        inverse_hyperbolic_cosine = ts.arccosh(arccosh_values)
        inverse_hyperbolic_tangent = ts.arctanh(arctanh_values)

        self.assertEqual(inverse_hyperbolic_cosine.shape, arccosh_values.shape)
        self.assertEqual(inverse_hyperbolic_tangent.shape, arctanh_values.shape)
        for actual, expected in zip(
            inverse_hyperbolic_cosine.tolist(),
            map(math.acosh, arccosh_values.tolist()),
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            inverse_hyperbolic_tangent.tolist(),
            map(math.atanh, arctanh_values.tolist()),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_public_math_namespace_exposes_functions_and_classes(self):
        cases = (
            (ts.math.sinh, ts.math.Sinh, 0.5, math.sinh),
            (ts.math.cosh, ts.math.Cosh, 0.5, math.cosh),
            (ts.math.arcsinh, ts.math.ArcSinh, 0.5, math.asinh),
            (ts.math.arccosh, ts.math.ArcCosh, 1.5, math.acosh),
            (ts.math.arctanh, ts.math.ArcTanh, 0.5, math.atanh),
        )
        for function, operation, point, reference in cases:
            with self.subTest(function=function.__name__):
                expected = reference(point)
                self.assertAlmostEqual(function([point]).item(), expected)
                self.assertAlmostEqual(
                    operation.forward(ts.Tensor([point])).item(),
                    expected,
                )

    def test_integer_inputs_promote_and_float_inputs_preserve_dtype(self):
        for function in (ts.sinh, ts.cosh, ts.arcsinh, ts.arctanh):
            self.assertIs(
                function(ts.Tensor([0], dtype=ts.int32)).dtype,
                ts.float64,
            )
            self.assertIs(
                function(ts.Tensor([0], dtype=ts.float32)).dtype,
                ts.float32,
            )
        self.assertIs(
            ts.arccosh(ts.Tensor([1], dtype=ts.int32)).dtype,
            ts.float64,
        )
        self.assertIs(
            ts.arccosh(ts.Tensor([1], dtype=ts.float32)).dtype,
            ts.float32,
        )

    def test_sinh_and_cosh_report_floating_point_overflow_as_infinity(self):
        value = ts.Variable([1000.0, -1000.0])

        hyperbolic_sine = ts.sinh(value)
        hyperbolic_cosine = ts.cosh(value)
        sine_gradient = ts.grad(hyperbolic_sine, value)
        cosine_gradient = ts.grad(hyperbolic_cosine, value)

        self.assertEqual(hyperbolic_sine.data.tolist(), [math.inf, -math.inf])
        self.assertEqual(hyperbolic_cosine.data.tolist(), [math.inf, math.inf])
        self.assertEqual(sine_gradient.tolist(), [math.inf, math.inf])
        self.assertEqual(cosine_gradient.tolist(), [math.inf, -math.inf])

    def test_arccosh_enforces_its_real_domain(self):
        for point in (-1.0, 0.0, 0.999):
            with self.subTest(point=point):
                with self.assertRaisesRegex(
                    ValueError,
                    "greater than or equal to 1",
                ):
                    ts.arccosh([point])

    def test_arccosh_value_exists_but_derivative_is_undefined_at_one(self):
        self.assertEqual(ts.arccosh([1.0]).item(), 0.0)

        value = ts.Variable([1.0])
        with self.assertRaisesRegex(ValueError, "undefined at 1"):
            ts.grad(ts.arccosh(value), value)

    def test_arctanh_enforces_its_open_real_domain(self):
        for point in (-2.0, -1.0, 1.0, 2.0):
            with self.subTest(point=point):
                with self.assertRaisesRegex(
                    ValueError,
                    "strictly between -1 and 1",
                ):
                    ts.arctanh([point])

    def test_first_derivatives(self):
        point = 0.4
        positive_point = 2.0
        value = ts.Variable([point])
        positive_value = ts.Variable([positive_point])

        self.assertAlmostEqual(
            ts.grad(ts.sinh(value), value).item(),
            math.cosh(point),
        )
        self.assertAlmostEqual(
            ts.grad(ts.cosh(value), value).item(),
            math.sinh(point),
        )
        self.assertAlmostEqual(
            ts.grad(ts.arcsinh(value), value).item(),
            1.0 / math.sqrt(1.0 + point ** 2.0),
        )
        self.assertAlmostEqual(
            ts.grad(ts.arccosh(positive_value), positive_value).item(),
            1.0 / math.sqrt(positive_point ** 2.0 - 1.0),
        )
        self.assertAlmostEqual(
            ts.grad(ts.arctanh(value), value).item(),
            1.0 / (1.0 - point ** 2.0),
        )

    def test_inverse_gradients_remain_finite_for_large_inputs(self):
        point = 1.0e308
        arcsinh_value = ts.Variable([point])
        arccosh_value = ts.Variable([point])

        arcsinh_gradient = ts.grad(ts.arcsinh(arcsinh_value), arcsinh_value)
        arccosh_gradient = ts.grad(ts.arccosh(arccosh_value), arccosh_value)

        self.assertTrue(
            math.isclose(arcsinh_gradient.item(), 1.0e-308, rel_tol=1.0e-15)
        )
        self.assertTrue(
            math.isclose(arccosh_gradient.item(), 1.0e-308, rel_tol=1.0e-15)
        )

    def test_second_derivatives(self):
        point = 0.4
        positive_point = 2.0
        value = ts.Variable([point])
        positive_value = ts.Variable([positive_point])

        sinh_first = ts.grad(ts.sinh(value), value, create_graph=True)
        cosh_first = ts.grad(ts.cosh(value), value, create_graph=True)
        arcsinh_first = ts.grad(ts.arcsinh(value), value, create_graph=True)
        arccosh_first = ts.grad(
            ts.arccosh(positive_value),
            positive_value,
            create_graph=True,
        )
        arctanh_first = ts.grad(ts.arctanh(value), value, create_graph=True)

        self.assertAlmostEqual(
            ts.grad(sinh_first, value).item(),
            math.sinh(point),
        )
        self.assertAlmostEqual(
            ts.grad(cosh_first, value).item(),
            math.cosh(point),
        )
        self.assertAlmostEqual(
            ts.grad(arcsinh_first, value).item(),
            -point / (1.0 + point ** 2.0) ** 1.5,
        )
        self.assertAlmostEqual(
            ts.grad(arccosh_first, positive_value).item(),
            -positive_point / (positive_point ** 2.0 - 1.0) ** 1.5,
        )
        self.assertAlmostEqual(
            ts.grad(arctanh_first, value).item(),
            2.0 * point / (1.0 - point ** 2.0) ** 2.0,
        )

    def test_arcsinh_second_derivative_is_valid_at_zero(self):
        value = ts.Variable([0.0])

        first = ts.grad(ts.arcsinh(value), value, create_graph=True)
        second = ts.grad(first, value)

        self.assertEqual(first.data.item(), 1.0)
        self.assertEqual(second.item(), 0.0)

    def test_computation_forward_replays_hyperbolic_nodes(self):
        value = ts.Variable([0.25])
        output = (
            ts.sinh(value)
            + ts.cosh(value)
            + ts.arcsinh(value)
            + ts.arccosh(value + 1.0)
            + ts.arctanh(value)
        )
        computation = ts.graph.Computation(output)
        value.data = ts.Tensor([0.5])

        replayed = computation.forward()

        expected = (
            math.sinh(0.5)
            + math.cosh(0.5)
            + math.asinh(0.5)
            + math.acosh(1.5)
            + math.atanh(0.5)
        )
        self.assertAlmostEqual(replayed.item(), expected)

    def test_functions_pass_gradcheck(self):
        ordinary_values = ts.Tensor([-0.7, 0.2, 0.8])
        positive_values = ts.Tensor([1.2, 2.0, 5.0])

        for function in (ts.sinh, ts.cosh, ts.arcsinh, ts.arctanh):
            with self.subTest(function=function.__name__):
                self.assertTrue(ts.gradcheck(function, ordinary_values))
        self.assertTrue(ts.gradcheck(ts.arccosh, positive_values))


if __name__ == "__main__":
    unittest.main()
