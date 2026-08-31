import unittest

import tensors as ts
from tensors.utils.shape import coordinates_to_index, index_to_coordinates


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


class CoordinateConversionTests(unittest.TestCase):
    def test_scalar_shape_contains_one_logical_element(self):
        self.assertEqual(index_to_coordinates(0, ()), ())
        self.assertEqual(coordinates_to_index((), ()), 0)

    def test_index_and_coordinates_round_trip(self):
        shape = ts.Shape(2, 3, 4)
        for index in range(shape.size):
            coordinates = index_to_coordinates(index, shape)
            self.assertEqual(coordinates_to_index(coordinates, shape), index)

    def test_flat_index_outside_shape_is_rejected(self):
        for index in (-1, 6):
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, "out of range"):
                    index_to_coordinates(index, (2, 3))

    def test_coordinate_rank_and_bounds_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Coordinate rank"):
            coordinates_to_index((1,), (2, 3))
        with self.assertRaisesRegex(IndexError, "out of range"):
            coordinates_to_index((2, 0), (2, 3))

    def test_flat_index_and_coordinates_require_integers(self):
        for index in (1.0, True):
            with self.subTest(index=index):
                with self.assertRaisesRegex(TypeError, "index must be an integer"):
                    index_to_coordinates(index, (2, 2))
        for coordinates in ((1.0, 0), (True, 0)):
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(TypeError, "only integers"):
                    coordinates_to_index(coordinates, (2, 2))


if __name__ == "__main__":
    unittest.main()
