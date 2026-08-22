import unittest

from tensors.utils.shape import (
    coordinates_to_index,
    index_to_coordinates,
    normalize_shape,
    row_major_strides,
    shape_size,
)


class ShapeUtilityTests(unittest.TestCase):
    def test_normalize_shape_returns_an_immutable_tuple(self):
        self.assertEqual(normalize_shape([2, 3]), (2, 3))

    def test_normalize_shape_accepts_a_generator(self):
        self.assertEqual(normalize_shape(value for value in [2, 3]), (2, 3))

    def test_normalize_shape_rejects_non_iterable_input(self):
        with self.assertRaisesRegex(TypeError, "shape must be an iterable"):
            normalize_shape(2)

    def test_shape_size_multiplies_dimensions(self):
        self.assertEqual(shape_size((2, 3, 4)), 24)

    def test_scalar_shape_contains_one_element(self):
        self.assertEqual(shape_size(()), 1)
        self.assertEqual(row_major_strides(()), ())
        self.assertEqual(index_to_coordinates(0, ()), ())
        self.assertEqual(coordinates_to_index((), ()), 0)

    def test_zero_dimension_produces_an_empty_shape(self):
        self.assertEqual(shape_size((2, 0, 3)), 0)
        self.assertEqual(row_major_strides((2, 0, 3)), (0, 3, 1))

    def test_row_major_strides_are_calculated_from_trailing_dimensions(self):
        self.assertEqual(row_major_strides((2, 3, 4)), (12, 4, 1))

    def test_index_and_coordinates_round_trip(self):
        shape = (2, 3, 4)

        for index in range(shape_size(shape)):
            coordinates = index_to_coordinates(index, shape)
            self.assertEqual(coordinates_to_index(coordinates, shape), index)

    def test_flat_index_outside_shape_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            index_to_coordinates(6, (2, 3))

    def test_negative_flat_index_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            index_to_coordinates(-1, (2, 3))

    def test_coordinate_rank_must_match_shape_rank(self):
        with self.assertRaisesRegex(ValueError, "Coordinate rank"):
            coordinates_to_index((1,), (2, 3))

    def test_coordinate_outside_dimension_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            coordinates_to_index((2, 0), (2, 3))

    def test_negative_shape_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            shape_size((2, -1))

    def test_boolean_shape_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            normalize_shape((True, 2))

    def test_non_integer_shape_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            normalize_shape((2.0, 3))

    def test_flat_index_must_be_an_integer(self):
        with self.assertRaisesRegex(TypeError, "index must be an integer"):
            index_to_coordinates(1.0, (2, 2))

        with self.assertRaisesRegex(TypeError, "index must be an integer"):
            index_to_coordinates(True, (2, 2))

    def test_coordinates_must_contain_integers(self):
        with self.assertRaisesRegex(TypeError, "only integers"):
            coordinates_to_index((1.0, 0), (2, 2))

        with self.assertRaisesRegex(TypeError, "only integers"):
            coordinates_to_index((True, 0), (2, 2))


if __name__ == "__main__":
    unittest.main()
