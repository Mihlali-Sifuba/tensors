import unittest

from tensors.utils.indexing import indices_to_flat_index


class IndexingUtilityTests(unittest.TestCase):
    def test_indices_are_converted_to_a_row_major_flat_index(self):
        self.assertEqual(indices_to_flat_index((1, 2, 3), (2, 3, 4)), 23)

    def test_negative_indices_are_normalized_per_dimension(self):
        self.assertEqual(indices_to_flat_index((-1, -2), (3, 4)), 10)

    def test_scalar_shape_accepts_an_empty_coordinate_tuple(self):
        self.assertEqual(indices_to_flat_index((), ()), 0)

    def test_index_rank_must_match_shape_rank(self):
        with self.assertRaisesRegex(IndexError, "Expected 2 indices, got 1"):
            indices_to_flat_index((1,), (2, 3))

    def test_boolean_indices_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "not bools"):
            indices_to_flat_index((True,), (2,))

    def test_non_integer_indices_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be integers"):
            indices_to_flat_index((1.0,), (2,))

    def test_positive_index_outside_a_dimension_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            indices_to_flat_index((2, 0), (2, 3))

    def test_negative_index_outside_a_dimension_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            indices_to_flat_index((-3,), (2,))

    def test_indexing_an_empty_dimension_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            indices_to_flat_index((0,), (0,))


if __name__ == "__main__":
    unittest.main()
