import unittest

import tensors as ts


class TensorIndexingTests(unittest.TestCase):
    def test_reverse_slice_uses_python_semantics(self):
        tensor = ts.Tensor([1, 2, 3, 4])

        self.assertEqual(tensor[::-1].tolist(), [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(tensor[3:0:-2].tolist(), [4.0, 2.0])

    def test_single_index_selects_first_dimension(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        self.assertEqual(tensor[0].shape, (2,))
        self.assertEqual(tensor[0].tolist(), [1.0, 2.0])

    def test_tuple_indices_return_scalar(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor[1, 2], 6.0)
        self.assertEqual(tensor[-1, -2], 5.0)

    def test_tuple_slice_selects_column(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        column = tensor[:, 1]

        self.assertEqual(column.shape, (2,))
        self.assertEqual(column.tolist(), [2.0, 5.0])

    def test_tuple_slice_selects_row_tail(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        row_tail = tensor[1, 1:]

        self.assertEqual(row_tail.shape, (2,))
        self.assertEqual(row_tail.tolist(), [5.0, 6.0])

    def test_too_many_indices_are_rejected(self):
        with self.assertRaisesRegex(IndexError, "Too many indices"):
            _ = ts.Tensor([[1, 2]])[0, 0, 0]

    def test_out_of_range_index_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "Index out of range"):
            _ = ts.Tensor([1, 2])[2]

    def test_unsupported_index_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unsupported index type"):
            _ = ts.Tensor([1, 2])["bad"]

    def test_setitem_updates_1d_and_nd_values(self):
        vector = ts.Tensor([1, 2, 3])
        vector[1] = 20
        self.assertEqual(vector.tolist(), [1.0, 20.0, 3.0])

        matrix = ts.Tensor([[1, 2], [3, 4]])
        matrix[1, 0] = 30
        self.assertEqual(matrix.tolist(), [1.0, 2.0, 30.0, 4.0])

    def test_setitem_rejects_single_index_for_nd_tensor(self):
        with self.assertRaisesRegex(ValueError, "Cannot assign"):
            ts.Tensor([[1, 2]])[0] = 99


if __name__ == "__main__":
    unittest.main()
