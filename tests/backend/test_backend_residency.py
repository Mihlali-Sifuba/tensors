"""The backend-residency invariant, validated independently of any operation.

`docs/backends.md`, *Storage residency and transfers*. One rule governs every
value crossing a selected-backend execution boundary:

    a Tensor or Storage value must already reside on the expected backend.

:func:`~tensors.backend.validation.validate_backend_residency` states that rule
once. It takes the values and the backend the caller already selected, and it
answers only that question: it does not select a backend, move storage, invoke
a kernel, or read a dtype, a shape or an element. A result is a resident value
like any other, so the same call checks storage coming back out of a kernel.
"""

import inspect
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import tensors as ts
from tensors.backend import config
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy


class TaggedStorage(Storage):
    """Storage that claims a chosen backend.

    An array backend's buffer is a native array, because the dispatcher lowers
    a resident Tensor by reshaping and casting that buffer directly. A Python
    claim keeps the dependency-free ``array``.
    """

    def __init__(self, kind, values, dtype=ts.float64):
        super().__init__(dtype)
        self.kind = kind
        if kind == "python":
            self._buffer = array(dtype.typecode, values)
        else:
            import numpy

            self._buffer = numpy.asarray(list(values), dtype=numpy.dtype(dtype.name))

    @property
    def buffer(self):
        return self._buffer

    def copy(self):
        return TaggedStorage(self.kind, self.buffer, self.dtype)


def tagged_tensor(kind, values):
    return ts.Tensor._from_owned_storage(
        TaggedStorage(kind, values), dtype=ts.float64, shape=(len(values),)
    )


class NotResident:
    """An unrelated object that happens to carry a ``kind`` attribute.

    Residency is decided by repository type, not by attribute probing, so this
    must be ignored rather than mistaken for backend-resident data.
    """

    kind = "numpy"
    backend_storage = "numpy"


class InterfaceTests(unittest.TestCase):
    """The validator asks one question and is told the answer's context."""

    def test_it_takes_values_and_an_explicit_expected_backend(self):
        parameters = list(
            inspect.signature(validate_backend_residency).parameters
        )
        self.assertEqual(parameters, ["values", "expected_backend"])

    def test_it_has_no_context_parameter(self):
        parameters = inspect.signature(validate_backend_residency).parameters
        for forbidden in ("context", "operation", "op", "name"):
            self.assertNotIn(forbidden, parameters)

    def test_it_returns_none(self):
        value = tagged_tensor("python", [1.0])
        self.assertIsNone(validate_backend_residency((value,), "python"))
        self.assertIsNone(validate_backend_residency((), "python"))

    def test_it_never_selects_a_backend(self):
        value = tagged_tensor("python", [1.0])
        with patch.object(
            config,
            "get_backend",
            side_effect=AssertionError("validation must not select a backend"),
        ):
            self.assertIsNone(validate_backend_residency((value,), "python"))

    def test_it_returns_neither_a_backend_nor_storage(self):
        storage = TaggedStorage("python", [1.0])
        returned = validate_backend_residency((storage,), "python")
        self.assertIsNone(returned)
        self.assertNotIsInstance(returned, Storage)
        self.assertNotIsInstance(returned, str)


class MatchingResidencyTests(BackendTestCase):
    """Values already on the expected backend pass."""

    def test_a_python_tensor_passes_for_python(self):
        with ts.use_backend("python"):
            value = ts.Tensor([1.0, 2.0])
        self.assertIsNone(validate_backend_residency((value,), "python"))

    @requires_numpy
    def test_a_numpy_tensor_passes_for_numpy(self):
        with ts.use_backend("numpy"):
            value = ts.Tensor([1.0, 2.0])
        self.assertIsNone(validate_backend_residency((value,), "numpy"))

    @requires_cuda
    def test_a_cuda_tensor_passes_for_cuda(self):
        with ts.use_backend("cuda"):
            value = ts.Tensor([1.0, 2.0])
        self.assertIsNone(validate_backend_residency((value,), "cuda"))

    def test_raw_storage_passes_for_its_own_backend(self):
        for kind in ts.available_backends():
            with self.subTest(backend=kind):
                storage = TaggedStorage(kind, [1.0])
                self.assertIsNone(validate_backend_residency((storage,), kind))

    def test_mixed_tensor_and_storage_values_pass_together(self):
        values = (
            tagged_tensor("python", [1.0]),
            TaggedStorage("python", [2.0]),
            tagged_tensor("python", [3.0]),
        )
        self.assertIsNone(validate_backend_residency(values, "python"))


class MismatchedResidencyTests(BackendTestCase):
    """A value resident elsewhere raises BackendMismatchError."""

    def test_a_python_tensor_is_refused_for_numpy(self):
        value = tagged_tensor("python", [1.0])
        with self.assertRaises(ts.BackendMismatchError):
            validate_backend_residency((value,), "numpy")

    def test_a_numpy_tensor_is_refused_for_python(self):
        value = tagged_tensor("numpy", [1.0])
        with self.assertRaises(ts.BackendMismatchError):
            validate_backend_residency((value,), "python")

    @requires_cuda
    def test_a_cuda_tensor_is_refused_for_python(self):
        with ts.use_backend("cuda"):
            value = ts.Tensor([1.0])
        with self.assertRaises(ts.BackendMismatchError):
            validate_backend_residency((value,), "python")

    def test_raw_storage_mismatches_raise(self):
        storage = TaggedStorage("cuda", [1.0])
        with self.assertRaises(ts.BackendMismatchError):
            validate_backend_residency((storage,), "python")

    def test_the_error_names_the_index_the_resident_and_the_expected(self):
        values = (
            tagged_tensor("python", [1.0]),
            TaggedStorage("cuda", [2.0]),
        )
        with self.assertRaises(ts.BackendMismatchError) as raised:
            validate_backend_residency(values, "python")
        message = str(raised.exception)
        self.assertIn("1", message)
        self.assertIn("cuda", message)
        self.assertIn("python", message)
        self.assertEqual(
            message,
            "Value 1 resides on the cuda backend, "
            "but the expected backend is python",
        )

    def test_the_first_failing_value_is_reported(self):
        values = (
            TaggedStorage("numpy", [1.0]),
            TaggedStorage("cuda", [2.0]),
        )
        with self.assertRaises(ts.BackendMismatchError) as raised:
            validate_backend_residency(values, "python")
        self.assertIn("Value 0", str(raised.exception))
        self.assertIn("numpy", str(raised.exception))

    def test_mixed_backend_operands_raise_before_any_kernel_runs(self):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        left = tagged_tensor("python", [1.0])
        right = tagged_tensor("cuda", [2.0])
        with patch.object(dispatch, "load_backend") as loader:
            with self.assertRaises(ts.BackendMismatchError):
                left + right
        loader.assert_not_called()


class NonResidentValueTests(BackendTestCase):
    """Values belonging to no backend are skipped, not rejected."""

    def test_scalars_shapes_and_dtypes_are_ignored(self):
        values = (2, 2.5, -1, (2, 3), ts.float32, ts.int64, "python", None)
        self.assertIsNone(validate_backend_residency(values, "numpy"))

    def test_an_unrelated_object_carrying_kind_is_ignored(self):
        """Residency is decided by type, never by attribute probing."""
        self.assertIsNone(
            validate_backend_residency((NotResident(),), "python")
        )

    def test_scalar_arithmetic_still_executes(self):
        value = ts.Tensor([1, 2], dtype=ts.int32)
        self.assertEqual((value + 2).tolist(), [3, 4])
        self.assertEqual((value * 2).tolist(), [2, 4])
        self.assertEqual((value - 2).tolist(), [-1, 0])

    def test_validating_a_scalar_creates_and_converts_no_storage(self):
        from tensors.backend import conversion

        with patch.object(
            conversion,
            "convert_storage",
            side_effect=AssertionError("validation must not convert storage"),
        ):
            with patch.object(
                TaggedStorage,
                "copy",
                side_effect=AssertionError("validation must not copy storage"),
            ):
                self.assertIsNone(
                    validate_backend_residency(
                        (tagged_tensor("python", [1.0]), 2, 2.5), "python"
                    )
                )


class ResultResidencyTests(BackendTestCase):
    """Storage returned by a kernel is checked by the same invariant."""

    def _dispatch_with(self, kernel):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        backend = SimpleNamespace(add=kernel)
        return patch(
            "tensors.backend.config.get_backend", return_value="numpy"
        ), patch.object(dispatch, "load_backend", return_value=backend)

    def test_a_result_on_the_selected_backend_is_accepted(self):
        selection, loader = self._dispatch_with(
            lambda *args, **kwargs: TaggedStorage("numpy", [3.0])
        )
        left = tagged_tensor("numpy", [1.0])
        with selection, loader:
            result = left + 2.0
        self.assertEqual(result.backend_storage.kind, "numpy")
        self.assertIsInstance(result, ts.Tensor)

    def test_a_result_from_another_backend_is_refused(self):
        selection, loader = self._dispatch_with(
            lambda *args, **kwargs: TaggedStorage("python", [3.0])
        )
        left = tagged_tensor("numpy", [1.0])
        with selection, loader:
            with self.assertRaises(ts.BackendMismatchError):
                left + 2.0

    def test_a_refused_result_is_never_wrapped_in_a_tensor(self):
        selection, loader = self._dispatch_with(
            lambda *args, **kwargs: TaggedStorage("python", [3.0])
        )
        left = tagged_tensor("numpy", [1.0])
        with patch.object(
            ts.Tensor,
            "_from_owned_storage",
            side_effect=AssertionError("a refused result must not be wrapped"),
        ):
            with selection, loader:
                with self.assertRaises(ts.BackendMismatchError):
                    left + 2.0

    def test_operands_and_results_report_the_same_way(self):
        selection, loader = self._dispatch_with(
            lambda *args, **kwargs: TaggedStorage("cuda", [3.0])
        )
        left = tagged_tensor("numpy", [1.0])
        with selection, loader:
            with self.assertRaises(ts.BackendMismatchError) as raised:
                left + 2.0
        self.assertEqual(
            str(raised.exception),
            "Value 0 resides on the cuda backend, "
            "but the expected backend is numpy",
        )


class DispatchBehaviourTests(BackendTestCase):
    """Selection stays at the boundary; the validator only validates."""

    OPERATIONS = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b,
    }

    def test_valid_operands_execute_normally_on_every_backend(self):
        for backend in ts.available_backends():
            for name, operation in self.OPERATIONS.items():
                with self.subTest(backend=backend, operation=name):
                    with ts.use_backend(backend):
                        try:
                            left = ts.Tensor([4.0, 6.0])
                            right = ts.Tensor([2.0, 3.0])
                            result = operation(left, right)
                        except ts.BackendOperationUnsupportedError:
                            continue
                        self.assertEqual(result.backend_storage.kind, backend)

    def test_each_dispatcher_reads_the_selection_once(self):
        for name, operation in self.OPERATIONS.items():
            with self.subTest(operation=name):
                real = config.get_backend
                calls = []

                def counted():
                    calls.append(None)
                    return real()

                left = ts.Tensor([4.0, 6.0])
                right = ts.Tensor([2.0, 3.0])
                with patch.object(config, "get_backend", counted):
                    operation(left, right)
                self.assertEqual(
                    len(calls), 1, f"{name} read the selection {len(calls)} times"
                )

    def test_a_mismatch_is_detected_before_the_kernel_is_loaded(self):
        from tensors.backend.dispatch.arithmetic import divide as dispatch

        left = tagged_tensor("numpy", [1.0])
        with patch.object(dispatch, "load_backend") as loader:
            with self.assertRaises(ts.BackendMismatchError):
                left / 2.0
        loader.assert_not_called()

    def test_no_implicit_conversion_is_introduced(self):
        from tensors.backend import conversion

        left = tagged_tensor("numpy", [1.0])
        with patch.object(
            conversion,
            "convert_storage",
            side_effect=AssertionError("dispatch must not convert storage"),
        ):
            with self.assertRaises(ts.BackendMismatchError):
                left + 2.0

    def test_no_fallback_to_another_backend_is_introduced(self):
        import sys

        from tensors.backend.dispatch.arithmetic import add as dispatch

        python_add = sys.modules[
            "tensors.backend.python.kernels.arithmetic.add"
        ]

        left = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(add=lambda *args, **kwargs: None)
        with patch(
            "tensors.backend.config.get_backend", return_value="numpy"
        ), patch.object(dispatch, "load_backend", return_value=backend):
            with patch.object(
                python_add,
                "add",
                side_effect=AssertionError("must not fall back to Python"),
            ):
                with self.assertRaises(ts.BackendOperationUnsupportedError):
                    left + 2.0

    def test_small_workloads_stay_on_the_selected_backend(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    try:
                        one = ts.Tensor([1.0])
                        result = one + one
                    except ts.BackendOperationUnsupportedError:
                        continue
                    self.assertEqual(result.backend_storage.kind, backend)

    def test_unsupported_operation_errors_are_unchanged(self):
        from tensors.backend.dispatch.arithmetic import add as dispatch

        left = tagged_tensor("numpy", [1.0])
        backend = SimpleNamespace(add=lambda *args, **kwargs: None)
        with patch(
            "tensors.backend.config.get_backend", return_value="numpy"
        ), patch.object(dispatch, "load_backend", return_value=backend):
            with self.assertRaisesRegex(
                ts.BackendOperationUnsupportedError,
                "numpy backend cannot execute add",
            ):
                left + 2.0


class ConstructionBoundaryTests(BackendTestCase):
    """Tensor construction uses the same invariant as dispatch."""

    def test_construction_from_a_foreign_tensor_is_refused(self):
        source = ts.Tensor([1.0])
        with patch("tensors.backend.config.get_backend", return_value="numpy"):
            with self.assertRaises(ts.BackendMismatchError):
                ts.Tensor(source)

    def test_construction_from_foreign_storage_is_refused(self):
        with self.assertRaises(ts.BackendMismatchError):
            ts.Tensor(TaggedStorage("numpy", [1.0]))

    def test_construction_from_resident_values_is_unaffected(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    self.assertEqual(ts.Tensor([1, 2], dtype=ts.int32).tolist(), [1, 2])
                    self.assertEqual(ts.Tensor(3.0).item(), 3.0)


if __name__ == "__main__":
    unittest.main()
