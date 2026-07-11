from array import array
import unittest

import tensors as ts


class TensorConstructionTests(unittest.TestCase):
    def test_ragged_nested_lists_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ragged"):
            ts.Tensor([[1], [2, 3], []])

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid shape"):
            ts.Tensor([1, 2], shape=(-1, 2))

    def test_shape_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Data size"):
            ts.Tensor([1, 2, 3], shape=(2, 2))

    def test_raw_array_preserves_dtype(self):
        tensor = ts.Tensor(array("f", [1.0, 2.0]))

        self.assertIs(tensor.dtype, ts.float32)
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_tensor_copy_preserves_shape_dtype_and_independent_storage(self):
        original = ts.Tensor([[1, 2], [3, 4]], dtype=ts.float32)

        copy = ts.Tensor(original)

        self.assertEqual(copy.shape, (2, 2))
        self.assertIs(copy.dtype, ts.float32)
        copy[0, 0] = 99
        self.assertEqual(original.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_explicit_dtype_converts_raw_array_input(self):
        tensor = ts.Tensor(array("i", [1, 2]), dtype=ts.float32)

        self.assertIs(tensor.dtype, ts.float32)
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_unknown_dtype_typecode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown typecode"):
            ts.Tensor([1, 2], dtype="z")

    def test_unsupported_input_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unsupported data type"):
            ts.Tensor((1, 2, 3))

    def test_dtype_no_longer_creates_storage_arrays(self):
        self.assertFalse(hasattr(ts.float64, "make_array"))

    def test_clone_creates_independent_copy(self):
        tensor = ts.Tensor([1, 2, 3])
        clone = tensor.clone()

        self.assertEqual(clone.tolist(), [1.0, 2.0, 3.0])
        clone._data[0] = 99
        self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0])

    def test_clone_preserves_dtype(self):
        tensor = ts.Tensor([1, 2], dtype=ts.float32)

        self.assertIs(tensor.clone().dtype, ts.float32)

    def test_astype_converts_dtype(self):
        tensor = ts.Tensor([1, 2, 3])

        self.assertIs(tensor.astype(ts.float32).dtype, ts.float32)
        self.assertIs(tensor.astype(ts.int32).dtype, ts.int32)
        self.assertEqual(tensor.astype(ts.int32).tolist(), [1, 2, 3])

    def test_astype_with_string_typecode(self):
        tensor = ts.Tensor([1, 2, 3])

        self.assertIs(tensor.astype("f").dtype, ts.float32)

    def test_len_returns_first_dimension(self):
        self.assertEqual(len(ts.Tensor([1, 2, 3])), 3)
        self.assertEqual(len(ts.Tensor([[1, 2], [3, 4], [5, 6]])), 3)

    def test_len_returns_zero_for_empty_1d_tensor(self):
        self.assertEqual(len(ts.Tensor([])), 0)

    def test_scalar_tensor_supports_item_and_numeric_formatting(self):
        value = ts.std(ts.Tensor([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(value.item(), 0.816496580927726)
        self.assertEqual(f"{value:.4f}", "0.8165")

    def test_item_rejects_multi_element_tensor(self):
        with self.assertRaisesRegex(ValueError, "one element"):
            ts.Tensor([1.0, 2.0]).item()


if __name__ == "__main__":
    unittest.main()
