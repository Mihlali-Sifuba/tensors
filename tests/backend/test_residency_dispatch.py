import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import tensors as ts
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency
from tests.backend._support import requires_cuda, requires_numpy


class TaggedStorage(Storage):
    """Dependency-free storage used to exercise dispatch validation."""

    def __init__(self, kind, values, dtype=ts.float64):
        super().__init__(dtype)
        self.kind = kind
        self._buffer = array(dtype.typecode, values)

    @property
    def buffer(self):
        return self._buffer

    def copy(self):
        return TaggedStorage(self.kind, self.buffer, self.dtype)


def prepared(left, right, *, dtype, output_shape):
    """Stand in for a backend's operand preparation, unchanged.

    These tests are about which package is loaded and what the dispatcher does
    with the result, not about conversion, so a stub backend declares the
    identity preparation and the kernel sees exactly what was passed in.
    """
    return left, right


def tagged_tensor(kind, values):
    return ts.Tensor._from_owned_storage(
        TaggedStorage(kind, values), dtype=ts.float64, shape=(len(values),)
    )


class ValidationInterfaceTests(unittest.TestCase):
    def test_operand_validation_uses_the_supplied_backend(self):
        value = tagged_tensor("python", [1.0])

        with patch(
            "tensors.backend.config.get_backend",
            side_effect=AssertionError("validation must not select a backend"),
        ):
            validated = validate_backend_residency((value,), "python")

        self.assertIsNone(validated)

    def test_the_mismatch_message_names_index_resident_and_expected(self):
        storage = TaggedStorage("numpy", [1.0])

        with self.assertRaises(ts.BackendMismatchError) as raised:
            validate_backend_residency((storage,), "python")

        self.assertEqual(
            str(raised.exception),
            "Value 0 resides on the numpy backend, "
            "but the expected backend is python",
        )

    def test_one_validator_covers_operands_and_results(self):
        """A result is a resident value, so both directions fail alike."""
        from tensors.backend.dispatch.arithmetic import add as dispatch

        operand_side = tagged_tensor("numpy", [1.0])
        with self.assertRaises(ts.BackendMismatchError) as operand:
            operand_side + 1.0

        result_side = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(
            add=lambda *args, **kwargs: TaggedStorage("cuda", [3.0]),
            prepare_binary_operands=prepared,
        )
        with patch(
            "tensors.backend.config.get_backend", return_value="numpy"
        ), patch.object(dispatch, "load_backend", return_value=backend):
            with self.assertRaises(ts.BackendMismatchError) as result:
                result_side + 1.0

        self.assertEqual(
            str(operand.exception),
            "Value 0 resides on the numpy backend, "
            "but the expected backend is python",
        )
        self.assertEqual(
            str(result.exception),
            "Value 0 resides on the cuda backend, "
            "but the expected backend is numpy",
        )


class ArithmeticResidencyTests(unittest.TestCase):
    def test_python_operands_execute_and_preserve_dtype(self):
        left = ts.Tensor([1, 2], dtype=ts.int16)
        right = ts.Tensor([3, 4], dtype=ts.int16)

        result = left + right

        self.assertIsInstance(result._storage, PythonStorage)
        self.assertIs(result.dtype, ts.int16)
        self.assertEqual(result.tolist(), [4, 6])

    def test_operand_backend_must_match_active_backend(self):
        value = tagged_tensor("numpy", [1.0])

        with self.assertRaisesRegex(
            ts.BackendMismatchError,
            "Value 0 resides on the numpy backend, "
            "but the expected backend is python",
        ):
            value + 1.0

    def test_tensor_operands_cannot_have_different_backends(self):
        left = tagged_tensor("python", [1.0])
        right = tagged_tensor("cuda", [2.0])

        with self.assertRaisesRegex(
            ts.BackendMismatchError,
            "Value 1 resides on the cuda backend, "
            "but the expected backend is python",
        ):
            left + right

    def test_small_selected_backend_workload_uses_selected_kernel(self):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        left = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(
            add=lambda *args, **kwargs: TaggedStorage("numpy", [3.0]),
            prepare_binary_operands=prepared,
        )
        with patch("tensors.backend.config.get_backend", return_value="numpy"), patch.object(
            dispatch, "load_backend", return_value=backend
        ) as load:
            result = left + 2.0

        load.assert_called_once_with("numpy")
        self.assertEqual(result._storage.kind, "numpy")

    def test_wrong_backend_result_is_rejected(self):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        left = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(
            add=lambda *args, **kwargs: PythonStorage.from_values(
                [3.0], ts.float64
            ),
            prepare_binary_operands=prepared,
        )
        with patch("tensors.backend.config.get_backend", return_value="numpy"), patch.object(
            dispatch, "load_backend", return_value=backend
        ):
            with self.assertRaisesRegex(
                ts.BackendMismatchError,
                "Value 0 resides on the python backend, "
                "but the expected backend is numpy",
            ):
                left + 2.0

    def test_unsupported_selected_backend_operation_raises(self):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        left = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(add=lambda *args, **kwargs: None, prepare_binary_operands=prepared)
        with patch("tensors.backend.config.get_backend", return_value="numpy"), patch.object(
            dispatch, "load_backend", return_value=backend
        ):
            with self.assertRaisesRegex(
                ts.BackendOperationUnsupportedError,
                "numpy backend cannot execute add",
            ):
                left + 2.0


class ResidencyBoundaryTests(unittest.TestCase):
    def test_tensor_copy_construction_rejects_active_backend_mismatch(self):
        source = ts.Tensor([1.0])

        with patch("tensors.backend.config.get_backend", return_value="numpy"):
            with self.assertRaisesRegex(
                ts.BackendMismatchError,
                "Value 0 resides on the python backend, "
                "but the expected backend is numpy",
            ):
                ts.Tensor(source)

    def test_storage_construction_rejects_active_backend_mismatch(self):
        source = TaggedStorage("numpy", [1.0])

        with self.assertRaisesRegex(
            ts.BackendMismatchError,
            "Value 0 resides on the numpy backend, "
            "but the expected backend is python",
        ):
            ts.Tensor(source)

    def test_clone_uses_source_backend_under_another_active_backend(self):
        source = ts.Tensor([1.0])

        with patch("tensors.backend.config.get_backend", return_value="numpy"):
            cloned = source.clone()

        self.assertEqual(cloned._storage.kind, "python")
        self.assertIsNot(cloned._storage, source._storage)
        self.assertEqual(cloned.tolist(), [1.0])

    def test_host_inspection_uses_resident_storage_under_another_backend(self):
        source = ts.Tensor([1.0, 2.0])

        with patch("tensors.backend.config.get_backend", return_value="numpy"):
            values = source.tolist()

        self.assertEqual(values, [1.0, 2.0])
        self.assertEqual(source._storage.kind, "python")


@requires_numpy
class NumPyConstructionResidencyTests(unittest.TestCase):
    def test_literal_and_small_creation_are_numpy_native(self):
        with ts.use_backend("numpy"):
            literal = ts.Tensor([1.0])
            created = ts.full((1,), 2.0)

        self.assertIsInstance(literal._storage, NumPyStorage)
        self.assertIsInstance(created._storage, NumPyStorage)

    def test_literal_construction_does_not_create_python_storage(self):
        with patch.object(
            PythonStorage,
            "from_values",
            side_effect=AssertionError("unexpected PythonStorage construction"),
        ):
            with ts.use_backend("numpy"):
                value = ts.Tensor([1, 2], dtype=ts.int32)

        self.assertIsInstance(value._storage, NumPyStorage)
        self.assertEqual(value.tolist(), [1, 2])

    def test_internal_scalar_construction_is_numpy_native(self):
        from tensors.shape import Shape

        with ts.use_backend("numpy"):
            value = ts.Tensor._from_values((3,), ts.int32, Shape())

        self.assertIsInstance(value._storage, NumPyStorage)
        self.assertEqual(value.item(), 3)


@requires_cuda
class CudaConstructionResidencyTests(unittest.TestCase):
    def test_literal_and_small_creation_are_cuda_native(self):
        with ts.use_backend("cuda"):
            literal = ts.Tensor([1.0])
            created = ts.full((1,), 2.0)

        self.assertIsInstance(literal._storage, CudaStorage)
        self.assertIsInstance(created._storage, CudaStorage)


if __name__ == "__main__":
    unittest.main()
