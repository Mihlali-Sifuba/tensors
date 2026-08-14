import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class TrigonometricTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_functions_return_elementwise_values_and_preserve_shape(self):
        values = ts.Tensor([[0.0, math.pi / 4.0], [math.pi / 2.0, math.pi]])

        sine = ts.sin(values)
        cosine = ts.cos(values)
        tangent = ts.tan(values[:1])

        self.assertEqual(sine.shape, values.shape)
        self.assertEqual(cosine.shape, values.shape)
        self.assertEqual(tangent.shape, (1, 2))
        for actual, expected in zip(sine.tolist(), map(math.sin, values.tolist())):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(cosine.tolist(), map(math.cos, values.tolist())):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            tangent.tolist(),
            map(math.tan, values[:1].tolist()),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_public_math_namespace_exposes_functions_and_classes(self):
        self.assertAlmostEqual(ts.math.sin([math.pi / 2.0]).item(), 1.0)
        self.assertAlmostEqual(ts.math.cos([0.0]).item(), 1.0)
        self.assertAlmostEqual(ts.math.tan([math.pi / 4.0]).item(), 1.0)
        self.assertAlmostEqual(
            ts.math.Sin.forward(ts.Tensor([math.pi / 2.0])).item(),
            1.0,
        )
        self.assertAlmostEqual(ts.math.Cos.forward(ts.Tensor([0.0])).item(), 1.0)
        self.assertAlmostEqual(
            ts.math.Tan.forward(ts.Tensor([math.pi / 4.0])).item(),
            1.0,
        )

    def test_integer_inputs_promote_and_float_inputs_preserve_dtype(self):
        for function in (ts.sin, ts.cos, ts.tan):
            self.assertIs(function(ts.Tensor([0], dtype=ts.int32)).dtype, ts.float64)
            self.assertIs(function(ts.Tensor([0], dtype=ts.float32)).dtype, ts.float32)

    def test_first_derivatives(self):
        point = 0.4
        value = ts.Variable([point])

        sine_gradient = ts.grad(ts.sin(value), value)
        cosine_gradient = ts.grad(ts.cos(value), value)
        tangent_gradient = ts.grad(ts.tan(value), value)

        self.assertAlmostEqual(sine_gradient.item(), math.cos(point))
        self.assertAlmostEqual(cosine_gradient.item(), -math.sin(point))
        self.assertAlmostEqual(
            tangent_gradient.item(),
            1.0 / math.cos(point) ** 2.0,
        )

    def test_second_derivatives(self):
        point = 0.4
        value = ts.Variable([point])

        sine_first = ts.grad(ts.sin(value), value, create_graph=True)
        cosine_first = ts.grad(ts.cos(value), value, create_graph=True)
        tangent_first = ts.grad(ts.tan(value), value, create_graph=True)

        self.assertAlmostEqual(ts.grad(sine_first, value).item(), -math.sin(point))
        self.assertAlmostEqual(ts.grad(cosine_first, value).item(), -math.cos(point))
        self.assertAlmostEqual(
            ts.grad(tangent_first, value).item(),
            2.0 * math.tan(point) / math.cos(point) ** 2.0,
        )

    def test_computation_forward_replays_trigonometric_nodes(self):
        value = ts.Variable([0.0])
        output = ts.sin(value) + ts.cos(value)
        computation = ts.graph.Computation(output)
        value.data = ts.Tensor([math.pi / 2.0])

        replayed = computation.forward()

        self.assertAlmostEqual(replayed.item(), 1.0)
        self.assertAlmostEqual(ts.grad(output, value).item(), -1.0)

    def test_composite_trigonometric_expression_passes_gradcheck(self):
        values = ts.Tensor([-0.7, 0.2, 0.8])

        self.assertTrue(
            ts.gradcheck(
                lambda value: ts.sin(value) + ts.cos(value) * ts.tan(value),
                values,
            )
        )


if __name__ == "__main__":
    unittest.main()
