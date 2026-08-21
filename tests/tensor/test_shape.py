import unittest

import tensors as ts


class TensorShapeTests(unittest.TestCase):
    def test_scalar_shape_can_be_requested_with_single_element(self):
        tensor = ts.Tensor([7.0], shape=())

        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.ndim, 0)
        self.assertEqual(tensor.size, 1)
        self.assertEqual(tensor.item(), 7.0)

    def test_len_rejects_scalar_shape(self):
        tensor = ts.Tensor([7.0], shape=())

        with self.assertRaisesRegex(TypeError, "0-dimensional"):
            len(tensor)

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
