"""Behavioral contracts for provider-bound arithmetic."""

import unittest
from unittest.mock import patch
import tensors as ts
from tensors.backend.dispatch.arithmetic import execute_add
from tensors.backend.loading import load_backend
from tensors.backend.numpy.storage import NumPyStorage
from tests.backend._support import requires_numpy


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

    def test_small_automatic_work_does_not_load_provider(self):
        # Automatic selection keeps the workload policy, because the Python
        # reference satisfies the same numerical contract.
        value = ts.Tensor([2.0])
        with (
            ts.use_backend("auto"),
            patch(
                "tensors.backend.dispatch.arithmetic._execution.load_backend"
            ) as loader,
        ):
            result = execute_add(value, 3.0, dtype=ts.float64, output_shape=(1,))
        self.assertEqual(list(result.buffer), [5.0])
        loader.assert_not_called()

    def test_nested_context_restores_arithmetic_provider(self):
        value = ts.Tensor([2.0] * 32)
        with ts.use_backend("numpy"):
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
            with ts.use_backend("python"):
                self.assertNotIsInstance((value + 1.0)._storage, NumPyStorage)
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
