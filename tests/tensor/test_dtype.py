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

    def test_dtype_reports_its_numeric_category(self):
        for dtype in (ts.float64, ts.float32):
            with self.subTest(dtype=dtype):
                self.assertEqual(dtype.kind, "floating")

        for dtype in (ts.int64, ts.int32, ts.int16, ts.int8, ts.uint8):
            with self.subTest(dtype=dtype):
                self.assertEqual(dtype.kind, "integer")

    def test_integer_scalar_operations_promote_to_represent_the_scalar(self):
        cases = (
            (ts.Tensor([0], dtype=ts.uint8), -1, ts.int16, [-1]),
            (ts.Tensor([0], dtype=ts.int8), 128, ts.int16, [128]),
            (ts.Tensor([0], dtype=ts.int16), 40_000, ts.int32, [40_000]),
            (
                ts.Tensor([0], dtype=ts.int64),
                2 ** 63,
                ts.float64,
                [float(2 ** 63)],
            ),
        )

        for tensor, scalar, expected_dtype, expected_values in cases:
            with self.subTest(dtype=tensor.dtype, scalar=scalar):
                result = tensor + scalar

                self.assertIs(result.dtype, expected_dtype)
                self.assertEqual(result.tolist(), expected_values)

    def test_representable_integer_scalar_preserves_dtype(self):
        result = ts.Tensor([1], dtype=ts.uint8) + 2

        self.assertIs(result.dtype, ts.uint8)
        self.assertEqual(result.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
