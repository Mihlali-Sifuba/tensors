import math
import unittest

import tensors as ts


class AbsoluteValueTests(unittest.TestCase):
    def test_abs_preserves_shape_and_dtype(self):
        value = ts.Tensor([[-2.0, 0.0], [3.0, -4.0]], dtype=ts.float32)

        result = abs(value)

        self.assertEqual(result.shape, value.shape)
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [2.0, 0.0, 3.0, 4.0])

    def test_abs_uses_zero_subgradient_at_zero(self):
        value = ts.Variable([-2.0, 0.0, 3.0])

        first = ts.grad(ts.sum(ts.abs(value)), value, create_graph=True)
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(second.tolist(), [0.0, 0.0, 0.0])

    def test_abs_propagates_nan_to_value_and_gradient(self):
        value = ts.Variable([math.nan])
        result = ts.abs(value)

        ts.backward(result)

        self.assertTrue(math.isnan(result.data.item()))
        self.assertTrue(math.isnan(value.grad.item()))

    def test_abs_rejects_higher_derivative_at_nan(self):
        value = ts.Variable([math.nan])

        with self.assertRaisesRegex(ValueError, "undefined at NaN"):
            ts.grad(ts.abs(value), value, create_graph=True)


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


class ClipTests(unittest.TestCase):
    def test_clip_supports_two_sided_and_one_sided_bounds(self):
        value = ts.Tensor([-2.0, 0.5, 3.0])

        self.assertEqual(ts.clip(value, 0.0, 1.0).tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(ts.clip(value, min_value=0.0).tolist(), [0.0, 0.5, 3.0])
        self.assertEqual(ts.clip(value, max_value=1.0).tolist(), [-2.0, 0.5, 1.0])

    def test_clip_validates_bounds(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            ts.clip([1.0])
        with self.assertRaisesRegex(ValueError, "greater"):
            ts.clip([1.0], 2.0, 1.0)
        with self.assertRaisesRegex(TypeError, "number"):
            ts.clip([1.0], "zero", 1.0)

    def test_clip_gradient_uses_zero_subgradient_at_boundaries(self):
        value = ts.Variable([-1.0, 0.0, 0.5, 1.0, 2.0])

        first = ts.grad(
            ts.sum(ts.clip(value, 0.0, 1.0)),
            value,
            create_graph=True,
        )
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(second.tolist(), [0.0] * 5)


class ArgExtremaTests(unittest.TestCase):
    def test_argmin_and_argmax_return_first_ties_as_int64(self):
        value = ts.Tensor([3.0, 1.0, 1.0, 4.0, 4.0])

        minimum = ts.argmin(value)
        maximum = ts.argmax(value)

        self.assertIs(minimum.dtype, ts.int64)
        self.assertIs(maximum.dtype, ts.int64)
        self.assertEqual(minimum.tolist(), [1])
        self.assertEqual(maximum.tolist(), [3])

    def test_argmin_and_argmax_are_axis_aware(self):
        value = ts.Tensor([[3.0, 1.0], [0.0, 4.0]])

        self.assertEqual(ts.argmin(value, axis=1).tolist(), [1, 0])
        self.assertEqual(ts.argmax(value, axis=0).tolist(), [0, 1])
        kept = ts.argmax(value, axis=1, keepdims=True)
        self.assertEqual(kept.shape, (2, 1))
        self.assertEqual(kept.tolist(), [0, 1])

    def test_arg_extrema_of_variable_are_nondifferentiable_tensors(self):
        value = ts.Variable([2.0, 1.0])

        result = ts.argmin(value)

        self.assertIsInstance(result, ts.Tensor)
        self.assertNotIsInstance(result, ts.Variable)

    def test_arg_extrema_reject_empty_inputs_and_invalid_axes(self):
        with self.assertRaisesRegex(ValueError, "empty tensor"):
            ts.argmax(ts.Tensor([]))
        with self.assertRaisesRegex(TypeError, "integer or None"):
            ts.argmin([1.0], axis=(0,))

    def test_arg_extrema_select_first_nan(self):
        value = ts.Tensor([2.0, math.nan, math.nan, 1.0])

        self.assertEqual(ts.argmin(value).tolist(), [1])
        self.assertEqual(ts.argmax(value).tolist(), [1])


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

        self.assertTrue(
            ts.gradcheck(lambda a, b: ts.maximum(a, b), (left, right))
        )
        self.assertTrue(
            ts.gradcheck(lambda a, b: ts.minimum(a, b), (left, right))
        )


if __name__ == "__main__":
    unittest.main()
