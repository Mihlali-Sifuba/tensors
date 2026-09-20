"""The arithmetic dispatchers route; they do not understand Tensors.

Three responsibilities meet on the way to a kernel, and this module pins the
line between them:

- the **operation layer** resolves dtype and shape and converts a scalar;
- the **preparation boundary** reads the selection, holds Tensor operands to
  the residency rule, and asks the selected backend for native operands;
- the **dispatcher** receives an execution-ready request and routes it.

A dispatcher that can still see a Tensor has not been separated, so these
tests check what it receives and what its module is even able to name.
"""

import inspect
import unittest

import tensors as ts
from tensors.backend import preparation
from tensors.backend.dispatch.arithmetic import add as add_dispatch
from tensors.backend.dispatch.arithmetic import divide as divide_dispatch
from tensors.backend.dispatch.arithmetic import multiply as multiply_dispatch
from tensors.backend.dispatch.arithmetic import power as power_dispatch
from tensors.backend.dispatch.arithmetic import subtract as subtract_dispatch
from tensors.backend.preparation import BinaryExecution, prepare_binary_execution
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy

DISPATCHERS = (
    ("add", add_dispatch, add_dispatch.execute_add),
    ("subtract", subtract_dispatch, subtract_dispatch.execute_subtract),
    ("multiply", multiply_dispatch, multiply_dispatch.execute_multiply),
    ("divide", divide_dispatch, divide_dispatch.execute_divide),
    ("power", power_dispatch, power_dispatch.execute_power),
)

#: Things a dispatcher must no longer be able to name.
TENSOR_VOCABULARY = (
    "Tensor",
    "Scalar",
    "convert_scalar",
    "broadcast",
    "prepare_binary_operands",
)


class DispatcherInterfaceTests(unittest.TestCase):
    """A dispatcher takes one execution-ready request and nothing else."""

    def test_each_dispatcher_takes_a_single_request(self):
        for name, _module, execute in DISPATCHERS:
            with self.subTest(operation=name):
                parameters = list(inspect.signature(execute).parameters)
                self.assertEqual(parameters, ["request"])

    def test_no_dispatcher_takes_operands_dtype_or_shape(self):
        for name, _module, execute in DISPATCHERS:
            with self.subTest(operation=name):
                parameters = inspect.signature(execute).parameters
                for forbidden in ("left", "right", "dtype", "output_shape"):
                    self.assertNotIn(forbidden, parameters)

    def test_no_dispatcher_module_names_tensor_semantics(self):
        """The vocabulary is the evidence: it cannot use what it cannot name."""
        for name, module, _execute in DISPATCHERS:
            source = inspect.getsource(module)
            for word in TENSOR_VOCABULARY:
                with self.subTest(operation=name, word=word):
                    self.assertNotIn(word, source)

    def test_no_dispatcher_reads_the_backend_selection(self):
        """The selection travels in the request; the dispatcher never re-reads it."""
        for name, module, _execute in DISPATCHERS:
            with self.subTest(operation=name):
                self.assertNotIn("get_backend", inspect.getsource(module))


class RequestTests(BackendTestCase):
    """The request is resolved: a backend, native operands, dtype and shape."""

    def test_a_request_carries_every_resolved_field(self):
        left = ts.Tensor([1.0, 2.0])
        right = ts.Tensor([3.0, 4.0])

        request = prepare_binary_execution(
            left, right, dtype=ts.float64, output_shape=(2,)
        )

        self.assertIsInstance(request, BinaryExecution)
        self.assertEqual(
            request._fields, ("backend", "left", "right", "dtype", "output_shape")
        )
        self.assertEqual(request.backend, ts.get_backend())
        self.assertIs(request.dtype, ts.float64)
        self.assertEqual(request.output_shape, (2,))

    def test_a_requests_operands_are_never_tensors(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    request = prepare_binary_execution(
                        ts.Tensor([1.0, 2.0]),
                        ts.Tensor([3.0, 4.0]),
                        dtype=ts.float64,
                        output_shape=(2,),
                    )
                self.assertNotIsInstance(request.left, ts.Tensor)
                self.assertNotIsInstance(request.right, ts.Tensor)

    def test_a_scalar_is_already_resolved_in_the_request(self):
        request = prepare_binary_execution(
            ts.Tensor([1.0, 2.0]), 3.0, dtype=ts.float64, output_shape=(2,)
        )
        self.assertNotIsInstance(request.left, ts.Tensor)
        self.assertNotIsInstance(request.right, ts.Tensor)

    def test_the_request_names_the_selected_backend(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    request = prepare_binary_execution(
                        ts.Tensor([1.0]), ts.Tensor([2.0]),
                        dtype=ts.float64, output_shape=(1,),
                    )
                self.assertEqual(request.backend, backend)

    def test_preparation_reads_the_selection_exactly_once(self):
        from unittest.mock import patch

        from tensors.backend import config

        real = config.get_backend
        reads = []

        def counted():
            reads.append(None)
            return real()

        left = ts.Tensor([1.0, 2.0])
        right = ts.Tensor([3.0, 4.0])
        with patch.object(config, "get_backend", counted):
            prepare_binary_execution(
                left, right, dtype=ts.float64, output_shape=(2,)
            )
        self.assertEqual(len(reads), 1)

    def test_a_whole_operation_still_reads_the_selection_once(self):
        """Preparation reads it; the dispatcher must not read it again."""
        from unittest.mock import patch

        from tensors.backend import config

        real = config.get_backend
        reads = []

        def counted():
            reads.append(None)
            return real()

        left = ts.Tensor([1.0, 2.0])
        right = ts.Tensor([3.0, 4.0])
        with patch.object(config, "get_backend", counted):
            left + right
        self.assertEqual(len(reads), 1)


class BackendSpecificPreparationTests(BackendTestCase):
    """Moving preparation behind a request preserved each backend's strategy."""

    def _request(self, backend, left, right, output_shape):
        with ts.use_backend(backend):
            return prepare_binary_execution(
                left, right, dtype=ts.float64, output_shape=output_shape
            )

    def test_python_still_expands_a_broadcast_operand(self):
        with ts.use_backend("python"):
            left = ts.Tensor([[1.0], [2.0]])
            right = ts.Tensor([10.0, 20.0])
        request = self._request("python", left, right, (2, 2))
        self.assertEqual(list(request.left), [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(list(request.right), [10.0, 20.0, 10.0, 20.0])

    def test_python_still_pairs_a_scalar_lazily(self):
        from itertools import islice, repeat

        with ts.use_backend("python"):
            tensor = ts.Tensor([1.0, 2.0, 3.0])
        request = self._request("python", tensor, 5.0, (3,))
        self.assertIsInstance(request.right, type(repeat(0)))
        self.assertEqual(list(islice(request.right, 3)), [5.0, 5.0, 5.0])

    @requires_numpy
    def test_numpy_still_hands_over_native_arrays(self):
        import numpy

        with ts.use_backend("numpy"):
            left = ts.Tensor([1.0, 2.0])
            right = ts.Tensor([3.0, 4.0])
        request = self._request("numpy", left, right, (2,))
        self.assertIsInstance(request.left, numpy.ndarray)
        self.assertEqual(request.left.dtype, numpy.dtype("float64"))

    @requires_cuda
    def test_cuda_operands_never_reach_the_host(self):
        import cupy

        with ts.use_backend("cuda"):
            left = ts.Tensor([1.0, 2.0])
            right = ts.Tensor([3.0, 4.0])
        request = self._request("cuda", left, right, (2,))
        for operand in (request.left, request.right):
            self.assertIsInstance(operand, cupy.ndarray)
            self.assertNotIsInstance(operand, list)


class ResidencyPlacementTests(BackendTestCase):
    """Operand residency is checked on Tensors; results on storage."""

    def test_a_foreign_operand_is_refused_by_preparation(self):
        from array import array

        from tensors.backend.storage import Storage

        class TaggedStorage(Storage):
            def __init__(self, kind, values, dtype=ts.float64):
                super().__init__(dtype)
                self.kind = kind
                self._buffer = array(dtype.typecode, values)

            @property
            def buffer(self):
                return self._buffer

            def copy(self):
                return TaggedStorage(self.kind, self.buffer, self.dtype)

        foreign = ts.Tensor._from_owned_storage(
            TaggedStorage("numpy", [1.0]), dtype=ts.float64, shape=(1,)
        )
        with self.assertRaises(ts.BackendMismatchError):
            prepare_binary_execution(
                foreign, 1.0, dtype=ts.float64, output_shape=(1,)
            )

    def test_a_foreign_operand_never_reaches_the_dispatcher(self):
        from unittest.mock import patch

        from array import array

        from tensors.backend.storage import Storage

        class TaggedStorage(Storage):
            def __init__(self, kind, values, dtype=ts.float64):
                super().__init__(dtype)
                self.kind = kind
                self._buffer = array(dtype.typecode, values)

            @property
            def buffer(self):
                return self._buffer

            def copy(self):
                return TaggedStorage(self.kind, self.buffer, self.dtype)

        foreign = ts.Tensor._from_owned_storage(
            TaggedStorage("numpy", [1.0]), dtype=ts.float64, shape=(1,)
        )
        with patch.object(add_dispatch, "load_backend") as loader:
            with self.assertRaises(ts.BackendMismatchError):
                foreign + 1.0
        loader.assert_not_called()

    def test_the_dispatcher_still_checks_the_result(self):
        """It receives storage, never a Tensor, and still enforces residency."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from array import array

        from tensors.backend.storage import Storage

        class TaggedStorage(Storage):
            def __init__(self, kind, values, dtype=ts.float64):
                super().__init__(dtype)
                self.kind = kind
                self._buffer = array(dtype.typecode, values)

            @property
            def buffer(self):
                return self._buffer

            def copy(self):
                return TaggedStorage(self.kind, self.buffer, self.dtype)

        stub = SimpleNamespace(
            add=lambda *args, **kwargs: TaggedStorage("cuda", [1.0])
        )
        with ts.use_backend("python"):
            request = prepare_binary_execution(
                ts.Tensor([1.0]), ts.Tensor([2.0]),
                dtype=ts.float64, output_shape=(1,),
            )
            with patch.object(add_dispatch, "load_backend", return_value=stub):
                with self.assertRaises(ts.BackendMismatchError):
                    add_dispatch.execute_add(request)


class PreservedBehaviourTests(BackendTestCase):
    """The new boundary changed no documented result or message."""

    def test_a_declining_kernel_keeps_its_message(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        stub = SimpleNamespace(add=lambda *args, **kwargs: None)
        with ts.use_backend("python"):
            request = prepare_binary_execution(
                ts.Tensor([1.0]), ts.Tensor([2.0]),
                dtype=ts.float64, output_shape=(1,),
            )
            with patch.object(add_dispatch, "load_backend", return_value=stub):
                with self.assertRaisesRegex(
                    ts.BackendOperationUnsupportedError,
                    "python backend cannot execute add at dtype float64",
                ):
                    add_dispatch.execute_add(request)

    def test_dtype_scalar_and_broadcast_behaviour_are_unchanged(self):
        integer = ts.Tensor([1, 2], dtype=ts.int32)
        self.assertIs((integer + 3).dtype, ts.int32)
        self.assertIs((integer + ts.Tensor([1], dtype=ts.int64)).dtype, ts.int64)
        self.assertIs((integer / integer).dtype, ts.float64)
        self.assertRaises(TypeError, lambda: integer + 3.5)
        self.assertRaises(TypeError, lambda: integer + True)
        broadcast = ts.Tensor([[1.0], [2.0]]) + ts.Tensor([10.0, 20.0])
        self.assertEqual(broadcast.shape, (2, 2))
        self.assertEqual(broadcast.tolist(), [11.0, 21.0, 12.0, 22.0])

    def test_every_backend_still_agrees_on_the_result(self):
        with ts.use_backend("python"):
            expected = (ts.Tensor([1.0, 2.0]) + ts.Tensor([3.0, 4.0])).tolist()
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    try:
                        produced = ts.Tensor([1.0, 2.0]) + ts.Tensor([3.0, 4.0])
                    except ts.BackendOperationUnsupportedError:
                        continue
                    self.assertEqual(produced.tolist(), expected)
                    self.assertEqual(produced._storage.kind, backend)


if __name__ == "__main__":
    unittest.main()
