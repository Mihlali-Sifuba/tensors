import unittest

import tensors as ts


class TensorDisplayTests(unittest.TestCase):
    def test_1d_repr_is_single_line(self):
        representation = repr(ts.Tensor([1, 2], dtype=ts.int32))

        self.assertEqual(
            representation,
            "Tensor([1, 2], shape=(2,), dtype='int32')",
        )

    def test_2d_repr_contains_rows(self):
        representation = repr(ts.Tensor([[1, 2], [3, 4]]))

        self.assertIn("[1.0 2.0]", representation)
        self.assertIn("[3.0 4.0]", representation)

    def test_scalar_shape_repr_is_plain_scalar_value(self):
        representation = repr(ts.Tensor([7.0], shape=()))

        self.assertEqual(representation, "7.0")

    def test_empty_1d_repr_shows_shape_and_dtype(self):
        representation = repr(ts.Tensor([]))

        self.assertEqual(representation, "Tensor([], shape=(0,), dtype='float64')")

    def test_tolist_returns_a_plain_list_copy(self):
        tensor = ts.Tensor([1, 2, 3])

        values = tensor.tolist()
        values[0] = 99

        self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0])

    def test_item_returns_python_number(self):
        self.assertIsInstance(ts.Tensor([1], dtype=ts.int32).item(), int)
        self.assertIsInstance(ts.Tensor([1.0], dtype=ts.float64).item(), float)

    def test_scalar_tensor_formats_without_tensor_repr(self):
        tensor = ts.Tensor([12.3456])

        self.assertEqual(f"{tensor:.2f}", "12.35")


if __name__ == "__main__":
    unittest.main()
