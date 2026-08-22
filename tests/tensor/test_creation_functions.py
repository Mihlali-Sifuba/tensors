import unittest

import tensors as ts


class TensorCreationTests(unittest.TestCase):
    def test_zeros_ones_and_full_create_requested_shapes(self):
        zeros = ts.zeros((2, 3))
        ones = ts.ones((2, 3), dtype=ts.float32)
        full = ts.full((2, 3), -4, dtype=ts.int32)

        self.assertEqual(zeros.tolist(), [0.0] * 6)
        self.assertEqual(ones.tolist(), [1.0] * 6)
        self.assertEqual(full.tolist(), [-4] * 6)
        self.assertEqual(zeros.shape, (2, 3))
        self.assertEqual(ones.shape, (2, 3))
        self.assertEqual(full.shape, (2, 3))
        self.assertIs(zeros.dtype, ts.float64)
        self.assertIs(ones.dtype, ts.float32)
        self.assertIs(full.dtype, ts.int32)

    def test_filled_constructors_support_scalar_and_empty_shapes(self):
        scalar = ts.zeros(())
        empty = ts.ones((2, 0, 3))

        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.tolist(), [0.0])
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.tolist(), [])

    def test_full_validates_its_fill_value(self):
        for value in (True, "three", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "fill_value"):
                    ts.full((2,), value)

    def test_eye_creates_square_and_rectangular_offset_diagonals(self):
        square = ts.eye(3)
        upper = ts.eye(2, 3, k=1, dtype=ts.int32)
        lower = ts.eye(3, 2, k=-1)

        self.assertEqual(square.tolist(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(upper.tolist(), [0, 1, 0, 0, 0, 1])
        self.assertEqual(lower.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        self.assertEqual(square.shape, (3, 3))
        self.assertEqual(upper.shape, (2, 3))
        self.assertEqual(lower.shape, (3, 2))

    def test_eye_validates_dimensions_and_diagonal(self):
        for arguments in ((-1,), (True,), (2, -1)):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    ts.eye(*arguments)
        with self.assertRaisesRegex(TypeError, "k must be an integer"):
            ts.eye(2, k=True)

    def test_arange_supports_stop_ranges_steps_and_descending_values(self):
        self.assertEqual(ts.arange(5).tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(ts.arange(2, 8, 2).tolist(), [2.0, 4.0, 6.0])
        self.assertEqual(ts.arange(5, -1, -2).tolist(), [5.0, 3.0, 1.0])
        self.assertEqual(ts.arange(0.0, 1.0, 0.25).tolist(), [0.0, 0.25, 0.5, 0.75])

    def test_arange_preserves_explicit_dtype_and_supports_empty_ranges(self):
        integer = ts.arange(1, 4, dtype=ts.int32)
        empty = ts.arange(4, 1)

        self.assertEqual(integer.tolist(), [1, 2, 3])
        self.assertIs(integer.dtype, ts.int32)
        self.assertEqual(empty.tolist(), [])
        self.assertEqual(empty.shape, (0,))

    def test_arange_validates_numbers_step_and_finite_values(self):
        with self.assertRaisesRegex(TypeError, "start"):
            ts.arange(True)
        with self.assertRaisesRegex(ValueError, "step cannot be zero"):
            ts.arange(0, 1, 0)
        with self.assertRaisesRegex(ValueError, "finite"):
            ts.arange(0.0, float("inf"))

    def test_linspace_includes_endpoints_and_supports_descending_values(self):
        ascending = ts.linspace(0.0, 1.0, 5)
        descending = ts.linspace(1.0, -1.0, 5)

        self.assertEqual(ascending.tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(descending.tolist(), [1.0, 0.5, 0.0, -0.5, -1.0])

    def test_linspace_handles_zero_one_and_large_symmetric_counts(self):
        empty = ts.linspace(2.0, 5.0, 0)
        singleton = ts.linspace(2.0, 5.0, 1)
        large = ts.linspace(-1.0e308, 1.0e308, 3)

        self.assertEqual(empty.tolist(), [])
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(singleton.tolist(), [2.0])
        self.assertEqual(large.tolist(), [-1.0e308, 0.0, 1.0e308])

    def test_linspace_preserves_explicit_dtype_and_validates_arguments(self):
        result = ts.linspace(0.0, 1.0, 3, dtype=ts.float32)

        self.assertIs(result.dtype, ts.float32)
        with self.assertRaisesRegex(TypeError, "count must be an integer"):
            ts.linspace(0.0, 1.0, True)
        with self.assertRaisesRegex(ValueError, "count must be non-negative"):
            ts.linspace(0.0, 1.0, -1)
        with self.assertRaisesRegex(ValueError, "finite"):
            ts.linspace(0.0, float("inf"), 3)

    def test_creation_namespace_exposes_all_constructors(self):
        self.assertEqual(ts.creation.zeros((2,)).tolist(), [0.0, 0.0])
        self.assertEqual(ts.creation.ones((2,)).tolist(), [1.0, 1.0])
        self.assertEqual(ts.creation.full((2,), 3).tolist(), [3.0, 3.0])
        self.assertEqual(ts.creation.eye(1).tolist(), [1.0])
        self.assertEqual(ts.creation.arange(2).tolist(), [0.0, 1.0])
        self.assertEqual(ts.creation.linspace(0.0, 1.0, 2).tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
