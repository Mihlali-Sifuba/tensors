import unittest

import tensors as ts
from tensors.utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
)
from tensors.utils.indexing import coordinates_to_storage_index


class LogicalCoordinateConversionTests(unittest.TestCase):
    def test_scalar_shape_contains_one_logical_element(self):
        self.assertEqual(linear_index_to_coordinates(0, ()), ())
        self.assertEqual(coordinates_to_linear_index((), ()), 0)

    def test_vectors_matrices_and_higher_rank_shapes_round_trip(self):
        for shape in (ts.Shape(5), ts.Shape(2, 3), ts.Shape(2, 3, 4)):
            with self.subTest(shape=shape):
                for index in range(shape.size):
                    coordinates = linear_index_to_coordinates(index, shape)
                    self.assertEqual(
                        coordinates_to_linear_index(coordinates, shape),
                        index,
                    )

    def test_logical_linear_and_physical_storage_indices_can_differ(self):
        shape = ts.Shape(3, 2)
        coordinates = (1, 1)

        self.assertEqual(
            coordinates_to_linear_index(coordinates, shape),
            3,
        )
        self.assertEqual(
            coordinates_to_storage_index(
                coordinates,
                shape,
                ts.Strides(1, 3),
            ),
            4,
        )

    def test_linear_index_outside_shape_is_rejected(self):
        for index in (-1, 6):
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, "out of range"):
                    linear_index_to_coordinates(index, (2, 3))

    def test_coordinate_rank_and_bounds_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Coordinate rank"):
            coordinates_to_linear_index((1,), (2, 3))
        with self.assertRaisesRegex(IndexError, "out of range"):
            coordinates_to_linear_index((2, 0), (2, 3))

    def test_linear_indices_and_coordinates_require_integers(self):
        for index in (1.0, True):
            with self.subTest(index=index):
                with self.assertRaisesRegex(TypeError, "index must be an integer"):
                    linear_index_to_coordinates(index, (2, 2))
        for coordinates in ((1.0, 0), (True, 0)):
            with self.subTest(coordinates=coordinates):
                with self.assertRaisesRegex(TypeError, "only integers"):
                    coordinates_to_linear_index(coordinates, (2, 2))


if __name__ == "__main__":
    unittest.main()
