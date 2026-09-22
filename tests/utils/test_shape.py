import unittest

import tensors as ts


class ShapeTests(unittest.TestCase):
    def test_scalar_shape_has_rank_zero_and_size_one(self):
        shape = ts.Shape()
        self.assertEqual((shape.rank, shape.size, tuple(shape)), (0, 1, ()))

    def test_vector_matrix_and_higher_rank_shapes(self):
        vector = ts.Shape(5)
        matrix = ts.Shape(2, 3)
        volume = ts.Shape(2, 3, 4)

        self.assertEqual((vector.rank, vector.size), (1, 5))
        self.assertEqual((matrix.rank, matrix.size), (2, 6))
        self.assertEqual((volume.rank, volume.size), (3, 24))
        self.assertEqual(volume[1], 3)
        self.assertEqual(list(volume), [2, 3, 4])

    def test_zero_dimensions_are_valid(self):
        shape = ts.Shape(2, 0, 4)
        self.assertEqual((shape.rank, shape.size), (3, 0))

    def test_shape_rejects_non_integer_dimensions_with_type_error(self):
        for dimensions in ((2.5, 3), (True, 3)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(TypeError, "dimensions must be integers"):
                    ts.Shape(*dimensions)

    def test_shape_rejects_negative_dimensions_with_value_error(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            ts.Shape(-1, 3)

    def test_shape_from_iterable_accepts_generators(self):
        shape = ts.Shape.from_iterable(value for value in (2, 3))

        self.assertEqual(shape, (2, 3))
        with self.assertRaisesRegex(TypeError, "shape must be an iterable"):
            ts.Shape.from_iterable(2)

    def test_shape_is_immutable_and_tuple_compatible(self):
        shape = ts.Shape(2, 3)

        self.assertEqual(shape, (2, 3))
        self.assertEqual((2, 3), shape)
        with self.assertRaises(TypeError):
            shape[0] = 4
        with self.assertRaises(AttributeError):
            shape.rank = 5

    def test_shape_integer_index_returns_int_and_slice_returns_shape(self):
        shape = ts.Shape(2, 3, 4)

        self.assertIs(type(shape[1]), int)
        self.assertEqual(shape[1], 3)
        self.assertIsInstance(shape[1:], ts.Shape)
        self.assertEqual(shape[1:], ts.Shape(3, 4))

    def test_shape_broadcast_with_returns_shape_for_compatible_dimensions(self):
        cases = (
            (ts.Shape(3, 1), ts.Shape(1, 4), ts.Shape(3, 4)),
            (ts.Shape(5, 1, 7), ts.Shape(3, 7), ts.Shape(5, 3, 7)),
            (ts.Shape(), ts.Shape(2, 3), ts.Shape(2, 3)),
            (ts.Shape(2, 3), ts.Shape(2, 3), ts.Shape(2, 3)),
            (ts.Shape(1, 3), ts.Shape(2, 1), ts.Shape(2, 3)),
            (ts.Shape(1, 0), ts.Shape(3, 1), ts.Shape(3, 0)),
        )

        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                result = left.broadcast_with(right)
                self.assertIsInstance(result, ts.Shape)
                self.assertEqual(result, expected)

    def test_shape_broadcast_with_accepts_an_iterable(self):
        result = ts.Shape(3, 1).broadcast_with(value for value in (1, 4))

        self.assertIsInstance(result, ts.Shape)
        self.assertEqual(result, ts.Shape(3, 4))

    def test_shape_broadcast_with_validates_iterable_dimensions(self):
        with self.assertRaisesRegex(TypeError, "dimensions must be integers"):
            ts.Shape(2, 3).broadcast_with((True, 3))
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            ts.Shape(2, 3).broadcast_with((-1, 3))

    def test_shape_broadcast_with_rejects_incompatible_dimensions(self):
        with self.assertRaisesRegex(ValueError, "cannot be broadcast"):
            ts.Shape(2, 3).broadcast_with(ts.Shape(2, 4))

    def test_shape_broadcast_with_does_not_mutate_inputs(self):
        left = ts.Shape(3, 1)
        right = ts.Shape(1, 4)

        left.broadcast_with(right)

        self.assertEqual(left, ts.Shape(3, 1))
        self.assertEqual(right, ts.Shape(1, 4))


class StridesTests(unittest.TestCase):
    def test_contiguous_vector_matrix_and_higher_rank_strides(self):
        self.assertEqual(ts.Strides.contiguous(ts.Shape(5)), (1,))
        self.assertEqual(ts.Strides.contiguous(ts.Shape(2, 3)), (3, 1))
        self.assertEqual(
            ts.Strides.contiguous(ts.Shape(2, 3, 4)),
            (12, 4, 1),
        )

    def test_scalar_and_zero_sized_contiguous_strides(self):
        self.assertEqual(ts.Strides.contiguous(ts.Shape()), ())
        self.assertEqual(
            ts.Strides.contiguous(ts.Shape(2, 0, 3)),
            (0, 3, 1),
        )

    def test_zero_and_negative_strides_are_valid(self):
        strides = ts.Strides(0, -1)
        self.assertEqual((strides, tuple(strides)), ((0, -1), (0, -1)))

    def test_strides_reject_non_integer_values(self):
        for values in ((True,), (1.0,), ("1",)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(TypeError, "only integers"):
                    ts.Strides(*values)

    def test_strides_from_iterable_validates_input(self):
        self.assertEqual(
            ts.Strides.from_iterable(value for value in (3, 1)),
            (3, 1),
        )
        with self.assertRaisesRegex(TypeError, "iterable of integers"):
            ts.Strides.from_iterable(1)

    def test_strides_are_immutable(self):
        strides = ts.Strides(3, 1)
        with self.assertRaises(TypeError):
            strides[0] = 1

    def test_strides_integer_index_returns_int_and_slice_returns_strides(self):
        strides = ts.Strides(12, 4, 1)

        self.assertIs(type(strides[1]), int)
        self.assertEqual(strides[1], 4)
        self.assertIsInstance(strides[1:], ts.Strides)
        self.assertEqual(strides[1:], ts.Strides(4, 1))


Shape = ts.Shape


class StretchedAxesTests(unittest.TestCase):
    """The inverse of ``broadcast_with``: which axes a broadcast stretched.

    Reverse-mode differentiation asks this to know which axes of a gradient to
    sum, but nothing here reads an element or mentions a gradient — it is the
    arithmetic of two shapes, and the answers below come from the broadcasting
    rules rather than from running a reduction.
    """

    def test_a_singleton_axis_is_stretched(self):
        self.assertEqual(Shape(2, 3).stretched_axes_from((1, 3)), (0,))
        self.assertEqual(Shape(2, 3).stretched_axes_from((2, 1)), (1,))
        self.assertEqual(Shape(2, 3).stretched_axes_from((1, 1)), (0, 1))

    def test_an_axis_the_operand_never_had_is_stretched(self):
        """Shapes align from the right, so a missing axis counts as one."""
        self.assertEqual(Shape(2, 3).stretched_axes_from((3,)), (0,))
        self.assertEqual(Shape(4, 2, 3).stretched_axes_from((3,)), (0, 1))
        self.assertEqual(Shape(2, 3).stretched_axes_from(()), (0, 1))

    def test_an_equal_shape_stretched_nothing(self):
        self.assertEqual(Shape(2, 3).stretched_axes_from((2, 3)), ())
        self.assertEqual(Shape(1, 3).stretched_axes_from((3,)), ())
        self.assertEqual(Shape().stretched_axes_from(()), ())

    def test_it_agrees_with_broadcast_with(self):
        """Every axis it names is one the forward broadcast changed."""
        pairs = (((1, 3), (2, 1)), ((2, 3), (3,)), ((4, 1, 3), (2, 3)), ((5,), (5,)))
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                common = Shape(*left).broadcast_with(right)
                for operand in (left, right):
                    stretched = common.stretched_axes_from(operand)
                    padded = (1,) * (common.rank - len(operand)) + tuple(operand)
                    for axis in range(common.rank):
                        changed = padded[axis] != common[axis]
                        self.assertEqual(axis in stretched, changed)

    def test_a_shape_that_did_not_broadcast_here_is_refused(self):
        with self.assertRaisesRegex(ValueError, "did not broadcast"):
            Shape(2, 3).stretched_axes_from((2, 3, 4))
        with self.assertRaisesRegex(ValueError, "did not broadcast"):
            Shape(2, 3).stretched_axes_from((2, 2))
        with self.assertRaisesRegex(ValueError, "did not broadcast"):
            Shape(2, 3).stretched_axes_from((3, 3))


if __name__ == "__main__":
    unittest.main()
