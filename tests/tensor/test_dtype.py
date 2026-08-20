import unittest
from array import array

import tensors as ts


class TensorDtypeTests(unittest.TestCase):
    def test_default_dtype_is_float64_for_python_values(self):
        self.assertIs(ts.Tensor([1, 2, 3]).dtype, ts.float64)
        self.assertIs(ts.Tensor(1).dtype, ts.float64)

    def test_dtype_can_be_requested_with_typecode(self):
        tensor = ts.Tensor([1, 2, 3], dtype="i")

        self.assertIs(tensor.dtype, ts.int32)
        self.assertEqual(tensor.tolist(), [1, 2, 3])

    def test_dtype_can_be_requested_with_human_readable_name(self):
        dtype_names = {
            "float64": ts.float64,
            "float32": ts.float32,
            "int64": ts.int64,
            "int32": ts.int32,
            "int16": ts.int16,
            "int8": ts.int8,
            "uint8": ts.uint8,
        }

        for name, expected_dtype in dtype_names.items():
            with self.subTest(name=name):
                tensor = ts.Tensor([1, 2, 3], dtype=name)

                self.assertIs(tensor.dtype, expected_dtype)

    def test_copy_can_convert_tensor_dtype(self):
        tensor = ts.Tensor([1, 2, 3], dtype=ts.int32)

        converted = ts.Tensor(tensor, dtype=ts.float32)

        self.assertIs(converted.dtype, ts.float32)
        self.assertEqual(converted.tolist(), [1.0, 2.0, 3.0])

    def test_array_typecode_maps_to_public_dtype(self):
        self.assertIs(ts.Tensor(array("b", [1, 2])).dtype, ts.int8)
        self.assertIs(ts.Tensor(array("B", [1, 2])).dtype, ts.uint8)
        self.assertIs(ts.Tensor(array("q", [1, 2])).dtype, ts.int64)

    def test_dtype_rejects_non_dtype_object(self):
        with self.assertRaisesRegex(TypeError, "dtype must be"):
            ts.Tensor([1, 2], dtype=object())

    def test_dtype_equality_supports_typecodes(self):
        self.assertEqual(ts.float32, "f")
        self.assertNotEqual(ts.float32, "d")


if __name__ == "__main__":
    unittest.main()
