"""Behaviour that holds across several operation domains.

A test whose single body asserts over more than one domain belongs here
rather than being filed arbitrarily under one of them or duplicated."""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class UnaryConventionTests(unittest.TestCase):

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_math_namespace_exposes_elementwise_functions(self):
        self.assertAlmostEqual(ts.math.sqrt([4.0]).item(), 2.0)
        self.assertAlmostEqual(ts.math.exp([0.0]).item(), 1.0)
        self.assertAlmostEqual(ts.math.log([math.e]).item(), 1.0)
        self.assertEqual(ts.math.relu([-1.0, 2.0]).tolist(), [0.0, 2.0])
        self.assertAlmostEqual(ts.math.sigmoid([0.0]).item(), 0.5)
        self.assertAlmostEqual(ts.math.tanh([1.0]).item(), math.tanh(1.0))
        self.assertAlmostEqual(ts.math.softplus([0.0]).item(), math.log(2.0))

    def test_math_namespace_exposes_elementwise_operation_classes(self):
        self.assertAlmostEqual(ts.math.Sqrt().forward(ts.Tensor([4.0])).item(), 2.0)
        self.assertAlmostEqual(ts.math.Exp().forward(ts.Tensor([0.0])).item(), 1.0)
        self.assertAlmostEqual(ts.math.Log().forward(ts.Tensor([math.e])).item(), 1.0)
        self.assertEqual(
            ts.math.ReLU().forward(ts.Tensor([-1.0, 2.0])).tolist(), [0.0, 2.0]
        )
        self.assertAlmostEqual(ts.math.Sigmoid().forward(ts.Tensor([0.0])).item(), 0.5)
        self.assertAlmostEqual(
            ts.math.Tanh().forward(ts.Tensor([1.0])).item(), math.tanh(1.0)
        )
        self.assertAlmostEqual(
            ts.math.Softplus().forward(ts.Tensor([0.0])).item(), math.log(2.0)
        )

    def test_integer_inputs_promote_exp_and_log_to_float64(self):
        self.assertIs(ts.sqrt(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.exp(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.log(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.sigmoid(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.tanh(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.softplus(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)

    def test_float32_inputs_preserve_dtype(self):
        self.assertIs(ts.sqrt(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.exp(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.log(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.sigmoid(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.tanh(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.softplus(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)


class NamespaceTests(unittest.TestCase):
    def test_math_namespace_exports_new_operation_classes(self):
        self.assertEqual(ts.math.Abs().forward(ts.Tensor([-2.0])).tolist(), [2.0])
        self.assertEqual(ts.math.Prod().forward(ts.Tensor([2.0, 3.0])).tolist(), [6.0])
        self.assertEqual(
            ts.math.Maximum().forward(ts.Tensor([1.0]), ts.Tensor([2.0])).tolist(),
            [2.0],
        )


class NewPrimitiveGradcheckTests(unittest.TestCase):
    def test_differentiable_primitives_pass_finite_difference_checks(self):
        checks = (
            (
                "abs",
                lambda value: ts.abs(value),
                ts.Tensor([-2.0, 3.0]),
            ),
            (
                "prod",
                lambda value: ts.prod(value),
                ts.Tensor([1.5, 2.0, 3.0]),
            ),
            (
                "clip",
                lambda value: ts.clip(value, -1.0, 1.0),
                ts.Tensor([-2.0, 0.25, 2.0]),
            ),
            (
                "where",
                lambda value: ts.where([1, 0], value, -value),
                ts.Tensor([1.0, 2.0]),
            ),
        )
        for name, function, value in checks:
            with self.subTest(operation=name):
                self.assertTrue(ts.gradcheck(function, value))

    def test_elementwise_extrema_pass_two_input_gradcheck_away_from_ties(self):
        left = ts.Tensor([1.0, 4.0])
        right = ts.Tensor([2.0, 3.0])

        self.assertTrue(ts.gradcheck(lambda a, b: ts.maximum(a, b), (left, right)))
        self.assertTrue(ts.gradcheck(lambda a, b: ts.minimum(a, b), (left, right)))


if __name__ == "__main__":
    unittest.main()
