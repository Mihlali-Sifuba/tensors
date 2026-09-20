"""Strict dispatch reaches every backend the same way, Python included.

`docs/backends.md`, *Execution requirements*. Under strict selected-backend
execution the selection names a kernel package and the operation names a kernel
inside it. Python is one of those packages, so a strict dispatcher must load it
through :func:`~tensors.backend.loading.load_backend` exactly as it loads NumPy
or CUDA, rather than importing a Python "reference" directly.

These tests state that contract — which package is loaded, and that the loaded
package's kernel is the one invoked — without asserting anything about how a
dispatcher is written internally.
"""

import inspect
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import tensors as ts
from tensors.backend import preparation
from tensors.backend import config
from tensors.backend.dispatch import _selected
from tensors.backend.dispatch.arithmetic import add as add_dispatch
from tensors.backend.dispatch.arithmetic import divide as divide_dispatch
from tensors.backend.dispatch.arithmetic import multiply as multiply_dispatch
from tensors.backend.dispatch.arithmetic import power as power_dispatch
from tensors.backend.dispatch.arithmetic import subtract as subtract_dispatch
from tensors.backend.dispatch.manipulation import cast as cast_dispatch
from tensors.backend.loading import load_backend
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy

#: Each strict arithmetic dispatcher, its kernel name, and how to trigger it.
ARITHMETIC = (
    ("add", add_dispatch, lambda a, b: a + b),
    ("subtract", subtract_dispatch, lambda a, b: a - b),
    ("multiply", multiply_dispatch, lambda a, b: a * b),
    ("divide", divide_dispatch, lambda a, b: a / b),
    ("power", power_dispatch, lambda a, b: a**b),
)


class TaggedStorage(Storage):
    """Dependency-free storage that claims a chosen backend."""

    def __init__(self, kind, values, dtype=ts.float64):
        super().__init__(dtype)
        self.kind = kind
        self._buffer = array(dtype.typecode, values)

    @property
    def buffer(self):
        return self._buffer

    def copy(self):
        return TaggedStorage(self.kind, self.buffer, self.dtype)


def tagged_tensor(kind, values):
    return ts.Tensor._from_owned_storage(
        TaggedStorage(kind, values), dtype=ts.float64, shape=(len(values),)
    )


def prepared(left, right, *, dtype, output_shape):
    """Stand in for a backend's operand preparation, unchanged.

    These tests are about which package is loaded and what the dispatcher does
    with the result, not about conversion, so a stub backend declares the
    identity preparation and the kernel sees exactly what was passed in.
    """
    return left, right


class RecordingLoader:
    """Stands in for ``load_backend`` and records the backend it was asked for."""

    def __init__(self, package=None):
        self.requested = []
        self._package = package

    def __call__(self, backend):
        self.requested.append(backend)
        return self._package if self._package is not None else load_backend(backend)


class SelectedPackageTests(BackendTestCase):
    """The selection names the package; the package supplies the kernel."""

    def test_python_arithmetic_loads_the_python_backend_package(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                loader = RecordingLoader()
                with ts.use_backend("python"):
                    left = ts.Tensor([4.0, 6.0])
                    right = ts.Tensor([2.0, 3.0])
                    with patch.object(dispatch, "load_backend", loader), patch.object(
                        preparation, "load_backend", loader
                    ):
                        operation(left, right)
                self.assertTrue(loader.requested, "no package was loaded")
                self.assertEqual(set(loader.requested), {"python"})

    def test_the_loaded_python_package_kernel_is_the_one_invoked(self):
        """No direct import can bypass the package the selection names."""
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                calls = []

                def kernel(*args, **kwargs):
                    calls.append((args, kwargs))
                    return PythonStorage.from_values([9.0, 9.0], ts.float64)

                stub = SimpleNamespace(**{name: kernel}, prepare_binary_operands=prepared)
                with ts.use_backend("python"):
                    left = ts.Tensor([4.0, 6.0])
                    right = ts.Tensor([2.0, 3.0])
                    with patch.object(
                        dispatch, "load_backend", RecordingLoader(stub)
                    ), patch.object(
                        preparation, "load_backend", RecordingLoader(stub)
                    ):
                        result = operation(left, right)
                self.assertEqual(len(calls), 1, f"{name} did not use the package")
                self.assertEqual(result.tolist(), [9.0, 9.0])

    def test_python_arithmetic_matches_the_package_kernel_result(self):
        """Routing through the package computes what the kernel computes."""
        with ts.use_backend("python"):
            left = ts.Tensor([4.0, 6.0])
            right = ts.Tensor([2.0, 3.0])
            self.assertEqual((left + right).tolist(), [6.0, 9.0])
            self.assertEqual((left - right).tolist(), [2.0, 3.0])
            self.assertEqual((left * right).tolist(), [8.0, 18.0])
            self.assertEqual((left / right).tolist(), [2.0, 2.0])

    @requires_numpy
    def test_numpy_arithmetic_loads_the_numpy_backend_package(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                loader = RecordingLoader()
                with ts.use_backend("numpy"):
                    left = ts.Tensor([4.0, 6.0])
                    right = ts.Tensor([2.0, 3.0])
                    with patch.object(dispatch, "load_backend", loader), patch.object(
                        preparation, "load_backend", loader
                    ):
                        operation(left, right)
                self.assertTrue(loader.requested, "no package was loaded")
                self.assertEqual(set(loader.requested), {"numpy"})

    @requires_cuda
    def test_cuda_arithmetic_loads_the_cuda_backend_package(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                loader = RecordingLoader()
                with ts.use_backend("cuda"):
                    try:
                        left = ts.Tensor([4.0, 6.0])
                        right = ts.Tensor([2.0, 3.0])
                        with patch.object(dispatch, "load_backend", loader), patch.object(
                        preparation, "load_backend", loader
                    ):
                            operation(left, right)
                    except ts.BackendOperationUnsupportedError:
                        pass
                self.assertTrue(loader.requested, "no package was loaded")
                self.assertEqual(set(loader.requested), {"cuda"})

    def test_the_strict_helper_takes_no_python_reference(self):
        """Its callers name an operation; they do not supply an implementation."""
        parameters = inspect.signature(
            _selected.run_on_selected_backend
        ).parameters
        self.assertNotIn("reference", parameters)
        self.assertEqual(list(parameters)[0], "operation")

    def test_the_strict_helper_loads_the_selected_package(self):
        loader = RecordingLoader()
        with ts.use_backend("python"):
            with patch.object(_selected, "load_backend", loader):
                ts.full((2,), 3.0)
        self.assertTrue(loader.requested, "no package was loaded")
        self.assertEqual(set(loader.requested), {"python"})

    def test_the_strict_helper_uses_the_loaded_package_kernel(self):
        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return PythonStorage.from_values([7.0, 7.0], ts.float64)

        stub = SimpleNamespace(full=kernel)
        with ts.use_backend("python"):
            with patch.object(_selected, "load_backend", RecordingLoader(stub)):
                result = ts.full((2,), 3.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.tolist(), [7.0, 7.0])


class CastDispatchTests(BackendTestCase):
    """Cast follows the same uniform loading path."""

    def test_python_cast_loads_the_python_backend_package(self):
        loader = RecordingLoader()
        with ts.use_backend("python"):
            value = ts.Tensor([1, 2], dtype=ts.int32)
            with patch.object(cast_dispatch, "load_backend", loader):
                result = value.astype(ts.float64)
        self.assertTrue(loader.requested, "no package was loaded")
        self.assertEqual(set(loader.requested), {"python"})
        self.assertIs(result.dtype, ts.float64)

    def test_cast_uses_the_loaded_package_kernel(self):
        calls = []

        def kernel(value, *, dtype):
            calls.append(dtype)
            return PythonStorage.from_values([5.0, 5.0], dtype)

        stub = SimpleNamespace(cast_tensor=kernel)
        with ts.use_backend("python"):
            value = ts.Tensor([1, 2], dtype=ts.int32)
            with patch.object(
                cast_dispatch, "load_backend", RecordingLoader(stub)
            ):
                result = value.astype(ts.float64)
        self.assertEqual(calls, [ts.float64])
        self.assertEqual(result.tolist(), [5.0, 5.0])

    @requires_numpy
    def test_numpy_cast_loads_the_numpy_backend_package(self):
        loader = RecordingLoader()
        with ts.use_backend("numpy"):
            value = ts.Tensor([1, 2], dtype=ts.int32)
            with patch.object(cast_dispatch, "load_backend", loader):
                value.astype(ts.float64)
        self.assertTrue(loader.requested, "no package was loaded")
        self.assertEqual(set(loader.requested), {"numpy"})


class PreservedContractTests(BackendTestCase):
    """Uniform loading must not weaken anything the strict contract promises."""

    def test_operand_mismatch_is_detected_before_the_package_loads(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                loader = RecordingLoader()
                left = tagged_tensor("numpy", [1.0])
                right = ts.Tensor([2.0])
                with patch.object(dispatch, "load_backend", loader), patch.object(
                        preparation, "load_backend", loader
                    ):
                    with self.assertRaises(ts.BackendMismatchError):
                        operation(left, right)
                self.assertEqual(loader.requested, [])

    def test_a_result_from_another_backend_is_still_rejected(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                stub = SimpleNamespace(**{name: lambda *a, **k: TaggedStorage("cuda", [1.0])}, prepare_binary_operands=prepared)
                with ts.use_backend("python"):
                    left = ts.Tensor([4.0])
                    right = ts.Tensor([2.0])
                    with patch.object(
                        dispatch, "load_backend", RecordingLoader(stub)
                    ), patch.object(
                        preparation, "load_backend", RecordingLoader(stub)
                    ):
                        with self.assertRaises(ts.BackendMismatchError):
                            operation(left, right)

    def test_a_declining_kernel_raises_without_falling_back(self):
        for name, dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                stub = SimpleNamespace(**{name: lambda *a, **k: None}, prepare_binary_operands=prepared)
                with ts.use_backend("python"):
                    left = ts.Tensor([4.0])
                    right = ts.Tensor([2.0])
                    with patch.object(
                        dispatch, "load_backend", RecordingLoader(stub)
                    ), patch.object(
                        preparation, "load_backend", RecordingLoader(stub)
                    ):
                        with self.assertRaises(
                            ts.BackendOperationUnsupportedError
                        ):
                            operation(left, right)

    def test_a_declining_kernel_keeps_its_informative_message(self):
        stub = SimpleNamespace(add=lambda *a, **k: None, prepare_binary_operands=prepared)
        with ts.use_backend("python"):
            left = ts.Tensor([4.0])
            with patch.object(
                add_dispatch, "load_backend", RecordingLoader(stub)
            ):
                with self.assertRaisesRegex(
                    ts.BackendOperationUnsupportedError,
                    "python backend cannot execute add at dtype float64",
                ):
                    left + 2.0

    def test_the_strict_helper_keeps_its_informative_message(self):
        stub = SimpleNamespace(full=lambda *a, **k: None)
        with ts.use_backend("python"):
            with patch.object(
                _selected, "load_backend", RecordingLoader(stub)
            ):
                with self.assertRaisesRegex(
                    ts.BackendOperationUnsupportedError,
                    "python backend cannot execute full",
                ):
                    ts.full((2,), 3.0)

    def test_no_operand_is_converted_implicitly(self):
        from tensors.backend import conversion

        left = tagged_tensor("numpy", [1.0])
        with patch.object(
            conversion,
            "convert_storage",
            side_effect=AssertionError("dispatch must not convert operands"),
        ):
            with self.assertRaises(ts.BackendMismatchError):
                left + 2.0


class SelectionReadOnceTests(BackendTestCase):
    """Selection is read once per operation, on every strict path."""

    def _count_selection_reads(self, call):
        real = config.get_backend
        reads = []

        def counted():
            reads.append(None)
            return real()

        with patch.object(config, "get_backend", counted):
            call()
        return len(reads)

    def test_each_arithmetic_dispatcher_reads_the_selection_once(self):
        left = ts.Tensor([4.0, 6.0])
        right = ts.Tensor([2.0, 3.0])
        for name, _dispatch, operation in ARITHMETIC:
            with self.subTest(operation=name):
                reads = self._count_selection_reads(
                    lambda: operation(left, right)
                )
                self.assertEqual(reads, 1, f"{name} read the selection {reads} times")

    def test_cast_reads_the_selection_once(self):
        value = ts.Tensor([1, 2], dtype=ts.int32)
        reads = self._count_selection_reads(lambda: value.astype(ts.float64))
        self.assertEqual(reads, 1)

    def test_the_strict_helper_reads_the_selection_once(self):
        reads = self._count_selection_reads(lambda: ts.full((2,), 3.0))
        self.assertEqual(reads, 1)


if __name__ == "__main__":
    unittest.main()
