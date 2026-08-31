import unittest

import tensors as ts
from tensors.utils.indexing import (
    coordinates_to_storage_index,
    tensor_indices_to_storage_index,
)


class IndexingUtilityTests(unittest.TestCase):
    def test_indices_map_to_contiguous_storage(self):
        self.assertEqual(
            tensor_indices_to_storage_index((1, 2, 3), ts.Shape(2, 3, 4)),
            23,
        )

    def test_negative_indices_are_normalized_per_dimension(self):
        self.assertEqual(
            tensor_indices_to_storage_index((-1, -2), ts.Shape(3, 4)),
            10,
        )

    def test_scalar_shape_accepts_empty_coordinates(self):
        self.assertEqual(tensor_indices_to_storage_index((), ts.Shape()), 0)

    def test_general_mapping_supports_nonzero_offset(self):
        self.assertEqual(
            coordinates_to_storage_index(
                (1, 2),
                ts.Shape(2, 3),
                ts.Strides(3, 1),
                offset=4,
            ),
            9,
        )

    def test_general_mapping_supports_zero_stride(self):
        self.assertEqual(
            coordinates_to_storage_index(
                (3, 2),
                ts.Shape(4, 3),
                ts.Strides(0, 1),
            ),
            2,
        )

    def test_general_mapping_supports_negative_stride(self):
        self.assertEqual(
            coordinates_to_storage_index(
                (2,),
                ts.Shape(3),
                ts.Strides(-1),
                offset=2,
            ),
            0,
        )

    def test_index_rank_must_match_shape_rank(self):
        with self.assertRaisesRegex(IndexError, "Expected 2 indices, got 1"):
            tensor_indices_to_storage_index((1,), ts.Shape(2, 3))

    def test_stride_rank_must_match_shape_rank(self):
        with self.assertRaisesRegex(ValueError, "Stride rank"):
            coordinates_to_storage_index(
                (1, 0),
                ts.Shape(2, 3),
                ts.Strides(1),
            )

    def test_coordinate_rank_must_match_shape_rank(self):
        with self.assertRaisesRegex(ValueError, "Coordinate rank"):
            coordinates_to_storage_index(
                (1,),
                ts.Shape(2, 3),
                ts.Strides(3, 1),
            )

    def test_boolean_and_non_integer_indices_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "not bools"):
            tensor_indices_to_storage_index((True,), ts.Shape(2))
        with self.assertRaisesRegex(TypeError, "must be integers"):
            tensor_indices_to_storage_index((1.0,), ts.Shape(2))

    def test_out_of_range_indices_are_rejected(self):
        for index in (2, -3):
            with self.subTest(index=index):
                with self.assertRaisesRegex(IndexError, "out of range"):
                    tensor_indices_to_storage_index((index,), ts.Shape(2))
        with self.assertRaisesRegex(IndexError, "out of range"):
            tensor_indices_to_storage_index((0,), ts.Shape(0))

    def test_offset_must_be_an_integer(self):
        with self.assertRaisesRegex(TypeError, "offset must be an integer"):
            coordinates_to_storage_index(
                (0,),
                ts.Shape(1),
                ts.Strides(1),
                offset=True,
            )


if __name__ == "__main__":
    unittest.main()
