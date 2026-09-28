import unittest
from unittest.mock import patch
import tensors as ts
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage
from tests.backend._support import requires_cuda, requires_numpy


@requires_numpy
class NumPyStorageTests(unittest.TestCase):
    """NumPy results retain native storage and cache their views."""

    def test_numpy_results_retain_native_storage(self):
        with ts.use_backend("numpy"):
            result = ts.full((64,), 2.0) + 3.0
        self.assertIsInstance(result.backend_storage, NumPyStorage)
        self.assertEqual(result.tolist(), [5.0] * 64)

    def test_backend_views_are_cached_until_mutation(self):
        import numpy

        value = ts.Tensor([1.0, 2.0, 3.0])
        original = value.backend_storage
        first = value._storage_for("numpy")
        second = value._storage_for("numpy")
        self.assertIs(first, second)
        self.assertIs(value.backend_storage, original)
        host_view = numpy.frombuffer(
            original.buffer, dtype=numpy.dtype(value.dtype.name)
        )
        self.assertTrue(numpy.shares_memory(first.buffer, host_view))
        value[0] = 4.0
        self.assertIsInstance(value.backend_storage, PythonStorage)
        self.assertEqual(set(value.backend_storage_cache), {"python"})


@requires_cuda
class CudaResidencyTests(unittest.TestCase):
    """Supported CUDA work stays on the device."""

    def test_floating_results_remain_device_resident(self):
        from tensors.backend.loading import load_backend

        backend = load_backend("cuda")
        with ts.use_backend("cuda"):
            value = ts.full((64,), 2.0)
            with patch.object(backend, "add", wraps=backend.add) as kernel:
                result = value * 3.0 + 1.0
        self.assertGreaterEqual(kernel.call_count, 1)
        self.assertIsInstance(value.backend_storage, CudaStorage)
        self.assertIsInstance(result.backend_storage, CudaStorage)
        self.assertEqual(result.tolist(), [7.0] * 64)

    def test_integer_operations_stay_device_resident(self):
        # Breaking change B12: CUDA integer arithmetic executes natively and
        # keeps its declared dtype instead of falling back to the host.
        with ts.use_backend("cuda"):
            result = ts.full((64,), 2, dtype=ts.int32) + 3
        self.assertIsInstance(result.backend_storage, CudaStorage)
        self.assertIs(result.dtype, ts.int32)
        self.assertEqual(result.tolist(), [5] * 64)

    def test_optimizer_updates_remain_device_resident(self):
        with ts.use_backend("cuda"):
            parameter = ts.Variable(ts.full((64,), 1.0))
            parameter.grad = ts.full((64,), 0.5)
            ts.optim.SGD([parameter], learning_rate=0.1).step()
        self.assertIsInstance(parameter.data.backend_storage, CudaStorage)
        self.assertAlmostEqual(parameter.data.tolist()[0], 0.95)


class BatchedOptimizerStorageTests(unittest.TestCase):
    """A batched optimizer update splits one result into per-parameter storage.

    Each parameter must receive its own backend's storage type, and the
    slices must stay independent: wrapping them in another backend's storage
    class, or handing two parameters the same buffer, both survive a values
    comparison and only show up as a type or aliasing check.
    """

    OPTIMIZERS = ("SGD", "Adam", "RMSprop")
    STORAGE_TYPES = {"numpy": NumPyStorage, "cuda": CudaStorage}

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _stepped_parameters(self, backend, name, count=4):
        with ts.use_backend(backend):
            parameters = []
            for index in range(count):
                parameter = ts.Variable(ts.full((64,), float(index + 1)))
                parameter.grad = ts.full((64,), 0.5 * (index + 1))
                parameters.append(parameter)
            getattr(ts.optim, name)(parameters, learning_rate=0.1).step()
        return parameters

    def test_every_parameter_keeps_its_own_backend_storage(self):
        for backend, storage_type in self.STORAGE_TYPES.items():
            if backend not in ts.available_backends():
                continue
            for name in self.OPTIMIZERS:
                with self.subTest(backend=backend, optimizer=name):
                    parameters = self._stepped_parameters(backend, name)
                    for index, parameter in enumerate(parameters):
                        self.assertIsInstance(
                            parameter.data.backend_storage,
                            storage_type,
                            f"parameter {index} left the {backend} backend",
                        )

    def test_batched_results_do_not_alias_each_other(self):
        for backend in self.STORAGE_TYPES:
            if backend not in ts.available_backends():
                continue
            for name in self.OPTIMIZERS:
                with self.subTest(backend=backend, optimizer=name):
                    parameters = self._stepped_parameters(backend, name)
                    buffers = [p.data.backend_storage.buffer for p in parameters]
                    for index, buffer in enumerate(buffers):
                        for other_index, other in enumerate(buffers[index + 1 :]):
                            self.assertIsNot(
                                buffer,
                                other,
                                f"parameters {index} and {other_index} share a buffer",
                            )
                    values = [p.data.tolist() for p in parameters]
                    for index, row in enumerate(values):
                        for other in values[index + 1 :]:
                            self.assertNotEqual(row, other)

    def test_batched_and_individual_updates_agree(self):
        entry_points = {
            "SGD": "execute_sgd_updates",
            "Adam": "execute_adam_updates",
            "RMSprop": "execute_rmsprop_updates",
        }
        for backend in self.STORAGE_TYPES:
            if backend not in ts.available_backends():
                continue
            for name in self.OPTIMIZERS:
                with self.subTest(backend=backend, optimizer=name):
                    batched = self._stepped_parameters(backend, name)
                    module = f"tensors.optim.{name.lower()}"
                    with patch(f"{module}.{entry_points[name]}", return_value=None):
                        individual = self._stepped_parameters(backend, name)
                    for left, right in zip(batched, individual):
                        for expected, actual in zip(
                            left.data.tolist(), right.data.tolist()
                        ):
                            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
