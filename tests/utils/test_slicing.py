import unittest

import tensors as ts
from tensors.utils.slicing import (
    flat_indices_from_ranges,
    slice_ranges_and_shape_from_key,
)


class SlicingUtilityTests(unittest.TestCase):
    def test_slice_key_produces_dimension_ranges_and_output_shape(self):
        ranges, result_shape = slice_ranges_and_shape_from_key(
            (1, slice(None, None, 2)),
            (3, 4, 2),
        )

        self.assertEqual([list(values) for values in ranges], [[1], [0, 2], [0, 1]])
        self.assertIsInstance(result_shape, ts.Shape)
        self.assertEqual(result_shape, (2, 2))

    def test_integer_dimensions_are_collapsed_from_the_output_shape(self):
        ranges, result_shape = slice_ranges_and_shape_from_key((1, 2), (3, 4))

        self.assertEqual([list(values) for values in ranges], [[1], [2]])
        self.assertIsInstance(result_shape, ts.Shape)
        self.assertEqual(result_shape, ())

    def test_tuple_shape_is_validated_as_shape_metadata(self):
        with self.assertRaisesRegex(TypeError, "dimensions must be integers"):
            slice_ranges_and_shape_from_key((slice(None),), (True,))
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            slice_ranges_and_shape_from_key((slice(None),), (-1,))

    def test_negative_integer_index_is_normalized(self):
        ranges, result_shape = slice_ranges_and_shape_from_key((-1,), (3, 2))

        self.assertEqual([list(values) for values in ranges], [[2], [0, 1]])
        self.assertEqual(result_shape, (2,))

    def test_reverse_slice_preserves_range_order(self):
        ranges, result_shape = slice_ranges_and_shape_from_key(
            (slice(None, None, -1),),
            (4,),
        )

        self.assertEqual(list(ranges[0]), [3, 2, 1, 0])
        self.assertEqual(result_shape, (4,))

    def test_empty_key_selects_every_dimension(self):
        ranges, result_shape = slice_ranges_and_shape_from_key((), (2, 3))

        self.assertEqual([list(values) for values in ranges], [[0, 1], [0, 1, 2]])
        self.assertEqual(result_shape, (2, 3))

    def test_too_many_indices_are_rejected(self):
        with self.assertRaisesRegex(IndexError, "Too many indices"):
            slice_ranges_and_shape_from_key((0, 0, 0), (2, 2))

    def test_boolean_indices_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "Boolean tensor indices"):
            slice_ranges_and_shape_from_key((True,), (2,))

    def test_unsupported_index_types_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unsupported index type"):
            slice_ranges_and_shape_from_key((1.0,), (2,))

    def test_integer_index_outside_a_dimension_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "out of range"):
            slice_ranges_and_shape_from_key((-3,), (2,))

    def test_flat_indices_follow_row_major_selection_order(self):
        result = flat_indices_from_ranges(
            [range(0, 2), range(1, 3)],
            (3, 1),
        )

        self.assertEqual(result, [1, 2, 4, 5])

    def test_flat_indices_preserve_reversed_range_order(self):
        result = flat_indices_from_ranges([range(2, -1, -1)], (1,))

        self.assertEqual(result, [2, 1, 0])

    def test_flat_indices_are_empty_when_any_range_is_empty(self):
        self.assertEqual(flat_indices_from_ranges([range(2), range(0)], (1, 1)), [])


if __name__ == "__main__":
    unittest.main()
