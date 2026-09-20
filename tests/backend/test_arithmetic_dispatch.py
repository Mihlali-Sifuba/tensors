"""Behavioral contracts for provider-bound arithmetic."""

import importlib
import unittest
from unittest.mock import patch
import tensors as ts
import tensors.backend as backend_state
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

#: Each dispatcher, and what ``2.0 <op> 2.0`` gives, so one loop covers all four.
DISPATCHERS = (
    ("add", execute_add, 4.0),
    ("subtract", execute_subtract, 0.0),
    ("multiply", execute_multiply, 4.0),
    ("divide", execute_divide, 1.0),
)


@requires_numpy
class ArithmeticDispatchTests(unittest.TestCase):

    def test_bound_provider_does_not_follow_later_context_changes(self):
        backend = load_backend("numpy")
        with ts.use_backend("numpy"):
            value = ts.Tensor([2.0])
        with ts.use_backend("python"):
            result = backend.add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(result.buffer.tolist(), [5.0])

    def test_each_dispatcher_reads_the_selection_once(self):
        """One lookup per call, in the dispatcher's own module.

        Each operation resolves the selection itself now, so the patch goes
        to the module that owns the entry point rather than to a shared one.
        """
        for name, execute, expected in DISPATCHERS:
            with self.subTest(operation=name):
                with ts.use_backend("numpy"):
                    value = ts.Tensor([2.0] * 32)
                    with patch(
                        "tensors.backend.config.get_backend",
                        wraps=config.get_backend,
                    ) as selection:
                        result = execute(
                            value, 2.0, dtype=ts.float64, output_shape=(32,)
                        )
                selection.assert_called_once_with()
                self.assertIsInstance(result, NumPyStorage)
                self.assertEqual(result.buffer.tolist(), [expected] * 32)

    def test_each_dispatcher_forwards_operands_dtype_and_shape(self):
        """The kernel receives exactly what the caller passed."""
        for name, execute, _ in DISPATCHERS:
            with self.subTest(operation=name):
                backend = load_backend("numpy")
                original = getattr(backend, name)
                with patch.object(backend, name, wraps=original) as kernel:
                    backend_state._clear_backend_kernel_cache()
                    with ts.use_backend("numpy"):
                        left = ts.Tensor([[2.0, 4.0], [6.0, 8.0]])
                        execute(left, 2.0, dtype=ts.float32, output_shape=(2, 2))
                backend_state._clear_backend_kernel_cache()
                kernel.assert_called_once()
                arguments, keywords = kernel.call_args
                self.assertIs(arguments[0], left)
                self.assertEqual(arguments[1], 2.0)
                self.assertIs(keywords["dtype"], ts.float32)
                self.assertEqual(keywords["output_shape"], (2, 2))

    def test_each_dispatcher_runs_on_the_selected_backend(self):
        """Python, NumPy and auto, at a size the old threshold sent away."""
        for selection, storage in (
            ("python", PythonStorage),
            ("numpy", NumPyStorage),
            ("auto", NumPyStorage),
        ):
            for name, execute, expected in DISPATCHERS:
                for size in (1, 4, 31, 64):
                    with self.subTest(selection=selection, operation=name, size=size):
                        with ts.use_backend(selection):
                            value = ts.Tensor([2.0] * size)
                            result = execute(
                                value, 2.0, dtype=ts.float64, output_shape=(size,)
                            )
                        self.assertIsInstance(result, storage)
                        self.assertEqual(list(result.buffer), [expected] * size)

    def test_no_dispatcher_calls_the_python_reference_under_numpy(self):
        """An accelerated selection must not reach the reference kernel."""
        import tensors.backend.python.kernels.arithmetic as python_kernels

        for name, execute, _ in DISPATCHERS:
            with self.subTest(operation=name):
                module = importlib.import_module(
                    f"tensors.backend.python.kernels.arithmetic.{name}"
                )
                with patch.object(
                    module, name, wraps=getattr(module, name)
                ) as reference:
                    with ts.use_backend("numpy"):
                        value = ts.Tensor([2.0] * 4)
                        execute(value, 2.0, dtype=ts.float64, output_shape=(4,))
                reference.assert_not_called()

    def test_each_declining_dispatcher_raises_without_falling_back(self):
        """A provider returning None is reported, never replaced."""
        for name, execute, _ in DISPATCHERS:
            with self.subTest(operation=name):
                backend = load_backend("numpy")
                with patch.object(backend, name, return_value=None):
                    backend_state._clear_backend_kernel_cache()
                    with ts.use_backend("numpy"):
                        value = ts.Tensor([2.0] * 4)
                        with self.assertRaises(
                            ts.BackendOperationUnsupportedError
                        ) as raised:
                            execute(value, 2.0, dtype=ts.float64, output_shape=(4,))
                    backend_state._clear_backend_kernel_cache()
                message = str(raised.exception)
                self.assertIn("numpy", message)
                self.assertIn(name, message)
                self.assertIn("float64", message)

    def test_small_explicit_numpy_work_still_loads_the_provider(self):
        # Breaking change B15. Explicit selection is an execution
        # requirement: workload size may decide how an operation runs, never
        # where. See docs/backends.md, Execution requirements.
        with ts.use_backend("numpy"):
            value = ts.Tensor([2.0])
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
        with ts.use_backend("auto"):
            value = ts.Tensor([2.0])
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
                    with ts.use_backend(selection):
                        value = ts.Tensor([2.0] * size)
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
        with ts.use_backend("numpy"):
            value = ts.Tensor([2.0] * 64)
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
        with ts.use_backend("auto"):
            value = ts.Tensor([2.0] * 64)
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

    def test_every_dispatcher_executes_on_cuda_including_small_work(self):
        """All four entry points, at sizes the old threshold sent away."""
        from tensors.backend.cuda.storage import CudaStorage

        for name, execute, expected in DISPATCHERS:
            for size in (1, 4, 31, 64):
                with self.subTest(operation=name, size=size):
                    with ts.use_backend("cuda"):
                        value = ts.Tensor([2.0] * size)
                        result = execute(
                            value, 2.0, dtype=ts.float64, output_shape=(size,)
                        )
                    self.assertIsInstance(result, CudaStorage)
                    self.assertEqual(result.buffer.tolist(), [expected] * size)

    def test_every_declining_cuda_kernel_raises(self):
        for name, execute, _ in DISPATCHERS:
            with self.subTest(operation=name):
                backend = load_backend("cuda")
                with patch.object(backend, name, return_value=None):
                    backend_state._clear_backend_kernel_cache()
                    with ts.use_backend("cuda"):
                        value = ts.Tensor([2.0] * 64)
                        with self.assertRaises(
                            ts.BackendOperationUnsupportedError
                        ) as raised:
                            execute(value, 2.0, dtype=ts.float64, output_shape=(64,))
                    backend_state._clear_backend_kernel_cache()
                message = str(raised.exception)
                self.assertIn("cuda", message)
                self.assertIn(name, message)

    def test_nested_context_restores_arithmetic_provider(self):
        with ts.use_backend("numpy"):
            value = ts.Tensor([2.0] * 32)
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
            with ts.use_backend("python"):
                with self.assertRaises(ts.BackendMismatchError):
                    value + 1.0
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
