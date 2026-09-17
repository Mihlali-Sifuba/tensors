"""Behavioral contracts for provider-bound arithmetic."""

import unittest
from unittest.mock import patch
import tensors as ts
from tensors.backend import config
from tensors.backend.dispatch.arithmetic import (
    execute_add,
    execute_divide,
    execute_multiply,
    execute_subtract,
)
from tensors.backend.loading import load_backend
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage
from tests.backend._support import requires_cuda, requires_numpy


@requires_numpy
class ArithmeticDispatchTests(unittest.TestCase):

    def test_bound_provider_does_not_follow_later_context_changes(self):
        backend = load_backend("numpy")
        value = ts.Tensor([2.0])
        with ts.use_backend("python"):
            result = backend.add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(result.buffer.tolist(), [5.0])

    def test_dispatch_reads_selection_once(self):
        value = ts.Tensor([2.0] * 32)
        with patch(
            "tensors.backend.dispatch.arithmetic._execution.get_backend",
            return_value="numpy",
        ) as selection:
            result = execute_add(value, 3.0, dtype=ts.float64, output_shape=(32,))
        selection.assert_called_once_with()
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(result.buffer.tolist(), [5.0] * 32)

    def test_small_explicit_numpy_work_still_loads_the_provider(self):
        # Breaking change B15. Explicit selection is an execution
        # requirement: workload size may decide how an operation runs, never
        # where. See docs/backends.md, Execution requirements.
        value = ts.Tensor([2.0])
        with ts.use_backend("numpy"):
            result = execute_add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(list(result.buffer), [5.0])

    def test_small_automatic_work_executes_on_numpy(self):
        """``auto`` resolves to NumPy and then behaves as NumPy.

        This replaces a test that asserted the opposite. The workload-size
        threshold used to send small arithmetic to Python even under
        ``auto``; where an operation runs is now decided by the selection
        alone. See docs/backends.md, Execution requirements.
        """
        value = ts.Tensor([2.0])
        with ts.use_backend("auto"):
            self.assertEqual(ts.get_backend(), "numpy")
            result = execute_add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(list(result.buffer), [5.0])

    def test_automatic_selection_without_numpy_executes_on_python(self):
        """With no NumPy installed, ``auto`` resolves to Python."""
        value = ts.Tensor([2.0])
        with patch.object(config, "_numpy_available", return_value=False):
            with ts.use_backend("auto"):
                self.assertEqual(ts.get_backend(), "python")
                result = execute_add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertIsInstance(result, PythonStorage)
        self.assertEqual(list(result.buffer), [5.0])

    def test_no_arithmetic_size_runs_on_another_backend(self):
        """Across four orders of magnitude, the selection alone decides."""
        for selection, expected in (
            ("python", PythonStorage),
            ("numpy", NumPyStorage),
            ("auto", NumPyStorage),
        ):
            for size in (1, 4, 31, 32, 1000):
                with self.subTest(selection=selection, size=size):
                    value = ts.Tensor([2.0] * size)
                    with ts.use_backend(selection):
                        for execute in (
                            execute_add,
                            execute_subtract,
                            execute_multiply,
                            execute_divide,
                        ):
                            result = execute(
                                value,
                                2.0,
                                dtype=ts.float64,
                                output_shape=(size,),
                            )
                            self.assertIsInstance(result, expected)

    def test_a_declining_kernel_raises_rather_than_falling_back(self):
        """An unsupported operation is reported, never quietly reassigned."""
        backend = load_backend("numpy")
        value = ts.Tensor([2.0] * 64)
        with ts.use_backend("numpy"):
            with patch.object(backend, "add", return_value=None) as declining:
                with self.assertRaises(ts.BackendOperationUnsupportedError) as raised:
                    execute_add(value, 3.0, dtype=ts.float64, output_shape=(64,))
        declining.assert_called_once()
        message = str(raised.exception)
        self.assertIn("numpy", message)
        self.assertIn("add", message)
        self.assertIn("float64", message)

    def test_a_declining_kernel_under_auto_also_raises(self):
        """``auto`` earns no fallback by being automatic."""
        backend = load_backend("numpy")
        value = ts.Tensor([2.0] * 64)
        with ts.use_backend("auto"):
            with patch.object(backend, "multiply", return_value=None):
                with self.assertRaises(ts.BackendOperationUnsupportedError):
                    execute_multiply(value, 3.0, dtype=ts.float64, output_shape=(64,))


@requires_cuda
class CudaArithmeticSelectionTests(unittest.TestCase):
    """Explicit CUDA selection executes on CUDA or says it cannot."""

    def test_arithmetic_executes_on_cuda_at_every_size(self):
        from tensors.backend.cuda.storage import CudaStorage

        for size in (1, 4, 31, 32, 1000):
            for dtype in (ts.float32, ts.float64, ts.int32, ts.uint8):
                with self.subTest(size=size, dtype=dtype.name):
                    with ts.use_backend("cuda"):
                        left = ts.full((size,), 2, dtype=dtype)
                        for operation in (
                            lambda a: a + a,
                            lambda a: a - a,
                            lambda a: a * a,
                        ):
                            self.assertIsInstance(operation(left)._storage, CudaStorage)

    def test_a_declining_cuda_kernel_raises(self):
        backend = load_backend("cuda")
        with ts.use_backend("cuda"):
            value = ts.full((64,), 2.0, dtype=ts.float64)
            with patch.object(backend, "subtract", return_value=None):
                with self.assertRaises(ts.BackendOperationUnsupportedError) as raised:
                    _ = value - value
        self.assertIn("cuda", str(raised.exception))

    def test_nested_context_restores_arithmetic_provider(self):
        value = ts.Tensor([2.0] * 32)
        with ts.use_backend("numpy"):
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
            with ts.use_backend("python"):
                self.assertNotIsInstance((value + 1.0)._storage, NumPyStorage)
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
