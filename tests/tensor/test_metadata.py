import importlib
import unittest

import tensors as ts
from tensors.storage import CudaStorage, NumPyStorage, PythonStorage


def synthetic_tensor(
    values,
    *,
    shape,
    strides,
    offset=0,
    dtype=ts.float64,
):
    storage = PythonStorage.from_values(values, dtype)
    return ts.Tensor._from_metadata(
        storage,
        shape=ts.Shape(*shape),
        strides=ts.Strides(*strides),
        offset=offset,
    )


class TensorMetadataTests(unittest.TestCase):
    def test_ordinary_tensor_has_explicit_contiguous_metadata(self):
        tensor = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        self.assertIsInstance(tensor.shape, ts.Shape)
        self.assertIsInstance(tensor.strides, ts.Strides)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.strides, (3, 1))
        self.assertEqual(tensor.offset, 0)
        self.assertTrue(tensor.is_contiguous)

    def test_singleton_dimension_strides_do_not_affect_contiguity(self):
        layouts = (
            ((1, 3), (100, 1), [1.0, 2.0, 3.0], True),
            ((2, 3), (3, 1), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], True),
            ((2, 3), (4, 1), [1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0], False),
        )

        for shape, strides, values, expected in layouts:
            with self.subTest(shape=shape, strides=strides):
                tensor = synthetic_tensor(
                    values,
                    shape=shape,
                    strides=strides,
                )
                self.assertIs(tensor.is_contiguous, expected)

    def test_scalar_and_zero_sized_tensor_metadata(self):
        scalar = ts.Tensor([7.0], shape=())
        empty = ts.Tensor([], shape=(2, 0, 3))

        self.assertEqual((scalar.shape, scalar.strides, scalar.offset), ((), (), 0))
        self.assertTrue(scalar.is_contiguous)
        self.assertEqual(empty.strides, (0, 3, 1))
        self.assertEqual(empty.size, 0)
        self.assertTrue(empty.is_contiguous)

    def test_zero_sized_layouts_are_contiguous_regardless_of_strides(self):
        for strides in ((0, 3, 1), (100, -7, 42)):
            with self.subTest(strides=strides):
                tensor = synthetic_tensor(
                    [],
                    shape=(2, 0, 3),
                    strides=strides,
                )
                self.assertEqual(tensor.size, 0)
                self.assertTrue(tensor.is_contiguous)

    def test_metadata_properties_are_read_only(self):
        tensor = ts.Tensor([[1.0]])

        with self.assertRaises(AttributeError):
            tensor.shape = ts.Shape(1)
        with self.assertRaises(AttributeError):
            tensor.strides = ts.Strides(1)
        with self.assertRaises(AttributeError):
            tensor.offset = 1
        with self.assertRaises(AttributeError):
            tensor.is_contiguous = False

    def test_contiguous_returns_self_for_contiguous_nonzero_offset(self):
        tensor = synthetic_tensor(
            [10.0, 20.0, 30.0],
            shape=(2,),
            strides=(1,),
            offset=1,
        )

        self.assertEqual(tensor[0], 20.0)
        self.assertEqual(tensor[1], 30.0)
        self.assertEqual(tensor.tolist(), [20.0, 30.0])
        self.assertTrue(tensor.is_contiguous)
        self.assertFalse(tensor._has_compact_storage)

        result = tensor.contiguous()

        self.assertIs(result, tensor)
        self.assertEqual(result.offset, 1)
        self.assertEqual(result._storage.size, 3)

    def test_storage_helpers_distinguish_physical_and_logical_order(self):
        tensor = synthetic_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            shape=(3, 2),
            strides=(1, 3),
        )

        physical = tensor._storage_for("python")
        logical = tensor._logical_storage_for("python")

        self.assertIsInstance(physical, PythonStorage)
        self.assertIsInstance(logical, PythonStorage)
        self.assertEqual(list(physical.buffer), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(
            list(tensor._logical_storage_indices()),
            [0, 3, 1, 4, 2, 5],
        )
        self.assertEqual(list(logical.buffer), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
        self.assertEqual(list(tensor._data), list(logical.buffer))
        self.assertEqual(tensor._value_at_storage_index(1), 1.0)
        self.assertFalse(tensor._has_compact_storage)

    def test_nonzero_offset_storage_helpers_preserve_index_spaces(self):
        tensor = synthetic_tensor(
            [10.0, 20.0, 30.0, 40.0],
            shape=(2,),
            strides=(1,),
            offset=1,
        )

        physical = tensor._storage_for("python")
        logical = tensor._logical_storage_for("python")

        self.assertIsInstance(physical, PythonStorage)
        self.assertIsInstance(logical, PythonStorage)
        self.assertEqual(list(physical.buffer), [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(list(tensor._logical_storage_indices()), [1, 2])
        self.assertEqual(list(logical.buffer), [20.0, 30.0])
        self.assertEqual(list(tensor._data), [20.0, 30.0])
        self.assertTrue(tensor.is_contiguous)
        self.assertFalse(tensor._has_compact_storage)

    def test_mutation_writes_to_physical_storage_index(self):
        tensor = synthetic_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            shape=(3, 2),
            strides=(1, 3),
        )

        tensor[1, 1] = 99.0

        storage = tensor._storage_for("python")
        self.assertIsInstance(storage, PythonStorage)
        self.assertEqual(
            list(storage.buffer),
            [0.0, 1.0, 2.0, 3.0, 99.0, 5.0],
        )
        self.assertEqual(tensor.tolist(), [0.0, 3.0, 1.0, 99.0, 2.0, 5.0])

    def test_zero_stride_repeats_physical_values(self):
        tensor = synthetic_tensor(
            [1.0, 2.0, 3.0],
            shape=(4, 3),
            strides=(0, 1),
        )

        self.assertEqual(
            tensor.tolist(),
            [1.0, 2.0, 3.0] * 4,
        )
        self.assertEqual(tensor[3, 2], 3.0)
        self.assertFalse(tensor.is_contiguous)

    def test_negative_stride_reverses_physical_traversal(self):
        tensor = synthetic_tensor(
            [1.0, 2.0, 3.0],
            shape=(3,),
            strides=(-1,),
            offset=2,
        )

        self.assertEqual(tensor.tolist(), [3.0, 2.0, 1.0])
        self.assertEqual(tensor[-1], 1.0)
        self.assertFalse(tensor.is_contiguous)

    def test_contiguous_materializes_non_contiguous_layout(self):
        tensor = synthetic_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            shape=(2, 2),
            strides=(3, 1),
            offset=1,
            dtype=ts.float32,
        )

        result = tensor.contiguous()

        self.assertIsNot(result, tensor)
        self.assertEqual(result.shape, tensor.shape)
        self.assertEqual(result.strides, (2, 1))
        self.assertEqual(result.offset, 0)
        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [1.0, 2.0, 4.0, 5.0])
        self.assertTrue(result.is_contiguous)
        result[0, 0] = 99.0
        self.assertEqual(tensor[0, 0], 1.0)

    def test_clone_and_dtype_conversion_materialize_logical_values(self):
        tensor = synthetic_tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            shape=(2, 2),
            strides=(3, 1),
            offset=1,
        )

        clone = tensor.clone()
        converted = tensor.astype(ts.float32)

        self.assertEqual(clone.tolist(), [1.0, 2.0, 4.0, 5.0])
        self.assertEqual(clone.strides, (2, 1))
        self.assertEqual(clone.offset, 0)
        self.assertEqual(converted.tolist(), clone.tolist())
        self.assertIs(converted.dtype, ts.float32)
        self.assertTrue(converted.is_contiguous)

    def test_internal_metadata_constructor_copies_storage(self):
        storage = PythonStorage.from_values([1.0, 2.0], ts.float64)
        tensor = ts.Tensor._from_metadata(
            storage,
            shape=ts.Shape(2),
            strides=ts.Strides(1),
        )

        storage.buffer[0] = 9.0

        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_invalid_layout_bounds_are_rejected(self):
        storage = PythonStorage.from_values([1.0, 2.0], ts.float64)

        with self.assertRaisesRegex(ValueError, "outside buffer"):
            ts.Tensor._from_metadata(
                storage,
                shape=ts.Shape(2),
                strides=ts.Strides(1),
                offset=1,
            )
        with self.assertRaisesRegex(ValueError, "Stride rank"):
            ts.Tensor._from_metadata(
                storage,
                shape=ts.Shape(2),
                strides=ts.Strides(),
            )

    def test_creation_and_layout_operations_return_compact_tensors(self):
        source = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        results = (
            ts.zeros((2, 3)),
            ts.ones((2, 3)),
            source.clone(),
            source.astype(ts.float32),
            ts.reshape(source, (4,)),
            ts.transpose(source),
            source[:, 1:],
            source + ts.Tensor([1.0, 2.0]),
        )

        for result in results:
            with self.subTest(shape=result.shape):
                self.assertEqual(result.offset, 0)
                self.assertEqual(
                    result.strides,
                    ts.Strides.contiguous(result.shape),
                )
                self.assertTrue(result.is_contiguous)


class BackendMetadataTests(unittest.TestCase):
    def test_numpy_contiguous_materialization_stays_numpy_native(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy backend is unavailable")
        numpy = importlib.import_module("numpy")
        storage = NumPyStorage(
            numpy.arange(40, dtype=numpy.float64),
            ts.float64,
        )
        tensor = ts.Tensor._from_metadata(
            storage,
            shape=ts.Shape(8, 4),
            strides=ts.Strides(5, 1),
            offset=1,
        )
        expected = [
            float(row * 5 + column + 1)
            for row in range(8)
            for column in range(4)
        ]

        result = tensor.contiguous()

        self.assertEqual(result._storage.kind, "numpy")
        self.assertEqual(result.tolist(), expected)
        with ts.use_backend("numpy"):
            calculated = tensor + 1.0
        self.assertEqual(calculated._storage.kind, "numpy")
        self.assertEqual(
            calculated.tolist(),
            [value + 1.0 for value in expected],
        )

    def test_cuda_contiguous_materialization_stays_device_native(self):
        if "cuda" not in ts.available_backends():
            self.skipTest("CUDA backend is unavailable")
        cupy = importlib.import_module("cupy")
        storage = CudaStorage(
            cupy.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            ts.float64,
        )
        tensor = ts.Tensor._from_metadata(
            storage,
            shape=ts.Shape(2, 2),
            strides=ts.Strides(3, 1),
            offset=1,
        )

        result = tensor.contiguous()

        self.assertEqual(result._storage.kind, "cuda")
        self.assertEqual(result.tolist(), [1.0, 2.0, 4.0, 5.0])
        with ts.use_backend("cuda"):
            calculated = tensor + 1.0
        self.assertEqual(calculated._storage.kind, "cuda")
        self.assertEqual(calculated.tolist(), [2.0, 3.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
