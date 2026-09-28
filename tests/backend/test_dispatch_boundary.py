"""The operation layer decides semantics; dispatch lowers and executes."""

import inspect
import unittest
from array import array
from itertools import islice, repeat
from types import SimpleNamespace
from unittest.mock import patch

import tensors as ts
from tensors.backend import config
from tensors.backend.dispatch.arithmetic import add as add_dispatch
from tensors.backend.dispatch.arithmetic import divide as divide_dispatch
from tensors.backend.dispatch.arithmetic import multiply as multiply_dispatch
from tensors.backend.dispatch.arithmetic import power as power_dispatch
from tensors.backend.dispatch.arithmetic import subtract as subtract_dispatch
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import convert_scalar
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy


DISPATCHERS = (
    ("add", add_dispatch, add_dispatch.execute_add),
    ("subtract", subtract_dispatch, subtract_dispatch.execute_subtract),
    ("multiply", multiply_dispatch, multiply_dispatch.execute_multiply),
    ("divide", divide_dispatch, divide_dispatch.execute_divide),
    ("power", power_dispatch, power_dispatch.execute_power),
)


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


def tagged_tensor(kind, values, dtype=ts.float64):
    return ts.Tensor._from_owned_storage(
        TaggedStorage(kind, values, dtype), dtype=dtype, shape=(len(values),)
    )


class DispatcherInterfaceTests(unittest.TestCase):
    def test_each_dispatcher_takes_operands_dtype_and_output_shape(self):
        expected = ["left", "right", "dtype", "output_shape"]
        for name, _module, execute in DISPATCHERS:
            with self.subTest(operation=name):
                self.assertEqual(list(inspect.signature(execute).parameters), expected)

    def test_request_abstraction_is_gone(self):
        for name, module, execute in DISPATCHERS:
            with self.subTest(operation=name):
                self.assertNotIn("request", inspect.signature(execute).parameters)
                source = inspect.getsource(module)
                self.assertNotIn("BinaryExecution", source)
                self.assertNotIn("prepare_binary_execution", source)
                self.assertNotIn("prepare_binary_operands", source)

    def test_dispatchers_do_not_resolve_dtype_or_convert_scalars(self):
        for name, module, _execute in DISPATCHERS:
            source = inspect.getsource(module)
            with self.subTest(operation=name):
                self.assertNotIn("resolve_result_dtype", source)
                self.assertNotIn("resolve_power", source)
                self.assertNotIn("convert_scalar", source)


class OperationBoundaryTests(BackendTestCase):
    def test_add_passes_tensor_semantics_to_dispatch(self):
        import importlib

        operation = importlib.import_module("tensors.operations.arithmetic.add")

        left = ts.Tensor([[1, 2]], dtype=ts.int32)
        right = ts.Tensor([[3], [4]], dtype=ts.int64)
        storage = PythonStorage.from_values([0, 0, 0, 0], ts.int64)
        with patch.object(operation, "execute_add", return_value=storage) as execute:
            result = operation.add(left, right)

        execute.assert_called_once_with(
            left, right, dtype=ts.int64, output_shape=(2, 2)
        )
        self.assertIs(result.dtype, ts.int64)
        self.assertEqual(result.shape, (2, 2))

    def test_add_converts_a_scalar_before_dispatch(self):
        import importlib

        operation = importlib.import_module("tensors.operations.arithmetic.add")

        left = ts.Tensor([1.0, 2.0], dtype=ts.float32)
        storage = PythonStorage.from_values([0.0, 0.0], ts.float32)
        with patch.object(operation, "execute_add", return_value=storage) as execute:
            operation.add(left, 0.1)

        execute.assert_called_once_with(
            left,
            convert_scalar(0.1, ts.float32),
            dtype=ts.float32,
            output_shape=(2,),
        )


class DispatcherExecutionTests(BackendTestCase):
    def test_each_dispatcher_reads_the_selection_once(self):
        left = ts.Tensor([4.0, 6.0])
        right = ts.Tensor([2.0, 3.0])
        for name, _module, execute in DISPATCHERS:
            reads = []
            real = config.get_backend

            def counted():
                reads.append(None)
                return real()

            with self.subTest(operation=name), patch.object(
                config, "get_backend", counted
            ):
                execute(left, right, dtype=ts.float64, output_shape=(2,))
            self.assertEqual(len(reads), 1)

    def test_operand_residency_is_checked_before_backend_loading(self):
        foreign = tagged_tensor("numpy", [1.0])
        with patch.object(add_dispatch, "load_backend") as loader:
            with self.assertRaises(ts.BackendMismatchError):
                add_dispatch.execute_add(
                    foreign, 1.0, dtype=ts.float64, output_shape=(1,)
                )
        loader.assert_not_called()

    def test_python_expands_tensor_operands_before_the_kernel(self):
        captured = []

        def kernel(left, right, **kwargs):
            captured.append((list(left), list(right), kwargs))
            return PythonStorage.from_values([0.0] * 4, ts.float64)

        backend = SimpleNamespace(add=kernel)
        left = ts.Tensor([[1.0], [2.0]])
        right = ts.Tensor([10.0, 20.0])
        with patch.object(add_dispatch, "load_backend", return_value=backend):
            add_dispatch.execute_add(
                left, right, dtype=ts.float64, output_shape=(2, 2)
            )

        self.assertEqual(captured[0][0], [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(captured[0][1], [10.0, 20.0, 10.0, 20.0])

    def test_python_pairs_a_scalar_lazily(self):
        captured = []

        def kernel(left, right, **kwargs):
            captured.append((left, right))
            return PythonStorage.from_values([0.0, 0.0, 0.0], ts.float64)

        backend = SimpleNamespace(add=kernel)
        left = ts.Tensor([1.0, 2.0, 3.0])
        with patch.object(add_dispatch, "load_backend", return_value=backend):
            add_dispatch.execute_add(
                left, 5.0, dtype=ts.float64, output_shape=(3,)
            )

        lowered_left, lowered_right = captured[0]
        self.assertEqual(list(lowered_left), [1.0, 2.0, 3.0])
        self.assertIsInstance(lowered_right, type(repeat(0)))
        self.assertEqual(list(islice(lowered_right, 3)), [5.0, 5.0, 5.0])

    @requires_numpy
    def test_numpy_receives_native_typed_operands_without_materialized_broadcast(self):
        import numpy

        captured = []

        def kernel(left, right, **kwargs):
            captured.append((left, right, kwargs))
            return TaggedStorage("numpy", [0.0] * 4, ts.float32)

        backend = SimpleNamespace(add=kernel)
        with ts.use_backend("numpy"):
            left = ts.Tensor([[1.0], [2.0]], dtype=ts.float64)
            right = ts.Tensor([10.0, 20.0], dtype=ts.float64)
            with patch.object(add_dispatch, "load_backend", return_value=backend):
                add_dispatch.execute_add(
                    left, right, dtype=ts.float32, output_shape=(2, 2)
                )

        lowered_left, lowered_right, _ = captured[0]
        self.assertIsInstance(lowered_left, numpy.ndarray)
        self.assertIsInstance(lowered_right, numpy.ndarray)
        self.assertEqual(lowered_left.dtype, numpy.dtype("float32"))
        self.assertEqual(lowered_left.shape, (2, 1))
        self.assertEqual(lowered_right.shape, (2,))

    @requires_cuda
    def test_cuda_receives_device_operands(self):
        import cupy

        captured = []
        with ts.use_backend("cuda"):
            left = ts.Tensor([1.0, 2.0])
            right = ts.Tensor([3.0, 4.0])
            backend = SimpleNamespace(
                add=lambda first, second, **kwargs: (
                    captured.append((first, second))
                    or TaggedStorage("cuda", [0.0, 0.0])
                )
            )
            with patch.object(add_dispatch, "load_backend", return_value=backend):
                add_dispatch.execute_add(
                    left, right, dtype=ts.float64, output_shape=(2,)
                )
        self.assertTrue(all(isinstance(value, cupy.ndarray) for value in captured[0]))

    def test_result_residency_is_checked(self):
        backend = SimpleNamespace(
            add=lambda *args, **kwargs: TaggedStorage("cuda", [3.0])
        )
        with patch.object(add_dispatch, "load_backend", return_value=backend):
            with self.assertRaises(ts.BackendMismatchError):
                add_dispatch.execute_add(
                    ts.Tensor([1.0]), 2.0, dtype=ts.float64, output_shape=(1,)
                )

    def test_a_declining_kernel_keeps_the_existing_error(self):
        backend = SimpleNamespace(add=lambda *args, **kwargs: None)
        with patch.object(add_dispatch, "load_backend", return_value=backend):
            with self.assertRaisesRegex(
                ts.BackendOperationUnsupportedError,
                "python backend cannot execute add at dtype float64",
            ):
                add_dispatch.execute_add(
                    ts.Tensor([1.0]), 2.0, dtype=ts.float64, output_shape=(1,)
                )


class PreservedBehaviourTests(BackendTestCase):
    def test_dtype_scalar_and_broadcast_rules_are_unchanged(self):
        integer = ts.Tensor([1, 2], dtype=ts.int32)
        self.assertIs((integer + 3).dtype, ts.int32)
        self.assertIs((integer + ts.Tensor([1], dtype=ts.int64)).dtype, ts.int64)
        self.assertIs((integer / integer).dtype, ts.float64)
        self.assertRaises(TypeError, lambda: integer + 3.5)
        broadcast = ts.Tensor([[1.0], [2.0]]) + ts.Tensor([10.0, 20.0])
        self.assertEqual(broadcast.shape, (2, 2))
        self.assertEqual(broadcast.tolist(), [11.0, 21.0, 12.0, 22.0])


if __name__ == "__main__":
    unittest.main()
