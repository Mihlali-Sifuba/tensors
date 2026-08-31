import unittest

import tensors as ts


class TensorShapeTests(unittest.TestCase):
    def test_python_scalar_infers_rank_zero_shape(self):
        tensor = ts.Tensor(7.0)

        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.ndim, 0)
        self.assertEqual(tensor.size, 1)
        self.assertEqual(tensor.item(), 7.0)

    def test_len_rejects_scalar_shape(self):
        tensor = ts.Tensor(7.0)

        with self.assertRaisesRegex(TypeError, "0-dimensional"):
            len(tensor)

    def test_scalar_indexing_uses_the_empty_coordinate(self):
        tensor = ts.Tensor(7.0)

        self.assertEqual(tensor[()], 7.0)
        with self.assertRaisesRegex(IndexError, "Too many indices"):
            tensor[0]

    def test_scalar_repr_and_item_expose_the_python_value(self):
        tensor = ts.Tensor(7.0)

        self.assertEqual(repr(tensor), "7.0")
        self.assertEqual(tensor.item(), 7.0)

    def test_scalar_arithmetic_preserves_rank_zero_and_broadcasts(self):
        scalar_result = ts.Tensor(7.0) + 2.0
        vector_result = ts.Tensor(7.0) + ts.Tensor([1.0, 2.0])

        self.assertEqual(scalar_result.shape, ())
        self.assertEqual(scalar_result.item(), 9.0)
        self.assertEqual(vector_result.shape, (2,))
        self.assertEqual(vector_result.tolist(), [8.0, 9.0])

    def test_scalar_dtype_conversion_preserves_rank_zero(self):
        converted = ts.Tensor(7.9).astype(ts.int32)

        self.assertEqual(converted.shape, ())
        self.assertEqual(converted.ndim, 0)
        self.assertIs(converted.dtype, ts.int32)
        self.assertEqual(converted.item(), 7)

    def test_scalar_arithmetic_preserves_shape_across_available_backends(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                result = ts.Tensor(7.0) + ts.Tensor(2.0)

            self.assertEqual(result.shape, ())
            self.assertEqual(result.item(), 9.0)
            self.assertIs(result.dtype, ts.float64)

    def test_single_element_sequence_remains_a_vector(self):
        tensor = ts.Tensor([7.0])

        self.assertEqual(tensor.shape, (1,))
        self.assertEqual(tensor.ndim, 1)

    def test_single_element_sequence_accepts_explicit_scalar_shape(self):
        tensor = ts.Tensor([7.0], shape=())

        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.ndim, 0)
        self.assertEqual(tensor.size, 1)
        self.assertEqual(tensor.item(), 7.0)

    def test_zero_sized_matrix_has_expected_size_and_rank(self):
        tensor = ts.Tensor([], shape=(0, 3))

        self.assertEqual(tensor.shape, (0, 3))
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.size, 0)

    def test_shape_accepts_any_iterable_of_dimensions(self):
        tensor = ts.Tensor([1, 2, 3, 4], shape=[2, 2])

        self.assertEqual(tensor.shape, (2, 2))

    def test_nested_list_infers_three_dimensions(self):
        tensor = ts.Tensor([[[1], [2]], [[3], [4]]])

        self.assertEqual(tensor.shape, (2, 2, 1))
        self.assertEqual(tensor.ndim, 3)
        self.assertEqual(tensor.size, 4)

    def test_deep_ragged_lists_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ragged"):
            ts.Tensor([[[1], [2, 3]], [[4], [5]]])


if __name__ == "__main__":
    unittest.main()
