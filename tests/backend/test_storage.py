import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.cuda as cuda_backend
from tensors.storage import CudaStorage, NumPyStorage, PythonStorage

from ._support import requires_cuda, requires_numpy


@requires_numpy
class NumPyStorageTests(unittest.TestCase):
    """NumPy results retain native storage and cache their views."""

    def test_numpy_results_retain_native_storage(self):
        with ts.use_backend("numpy"):
            result = ts.full((64,), 2.0) + 3.0

        self.assertIsInstance(result._storage, NumPyStorage)
        self.assertEqual(result.tolist(), [5.0] * 64)
    def test_backend_views_are_cached_until_mutation(self):
        import numpy

        value = ts.Tensor([1.0, 2.0, 3.0])
        original = value._storage

        first = value._storage_for("numpy")
        second = value._storage_for("numpy")

        self.assertIs(first, second)
        self.assertIs(value._storage, original)
        host_view = numpy.frombuffer(
            original.buffer,
            dtype=numpy.dtype(value.dtype.name),
        )
        self.assertTrue(numpy.shares_memory(first.buffer, host_view))
        value[0] = 4.0
        self.assertIsInstance(value._storage, PythonStorage)
        self.assertEqual(set(value._storage_cache), {"python"})


@requires_cuda
class CudaResidencyTests(unittest.TestCase):
    """Supported CUDA work stays on the device."""

    def test_floating_results_remain_device_resident(self):
        with ts.use_backend("cuda"):
            value = ts.full((64,), 2.0)
            with patch.object(
                cuda_backend,
                "binary",
                wraps=cuda_backend.binary,
            ) as kernel:
                result = value * 3.0 + 1.0

        self.assertGreaterEqual(kernel.call_count, 1)
        self.assertIsInstance(value._storage, CudaStorage)
        self.assertIsInstance(result._storage, CudaStorage)
        self.assertEqual(result.tolist(), [7.0] * 64)
    def test_integer_operations_use_reference_storage(self):
        with ts.use_backend("cuda"):
            result = ts.full((64,), 2, dtype=ts.int32) + 3

        self.assertIsInstance(result._storage, PythonStorage)
        self.assertEqual(result.tolist(), [5] * 64)
    def test_optimizer_updates_remain_device_resident(self):
        with ts.use_backend("cuda"):
            parameter = ts.Variable(ts.full((64,), 1.0))
            parameter.grad = ts.full((64,), 0.5)
            ts.optim.SGD([parameter], learning_rate=0.1).step()

        self.assertIsInstance(parameter.data._storage, CudaStorage)
        self.assertAlmostEqual(parameter.data[0], 0.95)


if __name__ == "__main__":
    unittest.main()
