"""Behavioral contracts for provider-bound arithmetic."""

import unittest
from unittest.mock import patch

import tensors as ts
from tensors.backend.dispatch.arithmetic import execute_add
from tensors.backend.loading import _load_array_backend
from tensors.backend.storage import NumPyStorage

from ._support import requires_numpy


@requires_numpy
class ArithmeticDispatchTests(unittest.TestCase):
    def test_bound_provider_does_not_follow_later_context_changes(self):
        backend = _load_array_backend("numpy")
        value = ts.Tensor([2.0])
        with ts.use_backend("python"):
            result = backend.add(
                value, 3.0, dtype=ts.float64, output_shape=(1,),
            )
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(result.buffer.tolist(), [5.0])

    def test_dispatch_reads_selection_once(self):
        value = ts.Tensor([2.0] * 32)
        with patch(
            "tensors.backend.dispatch.arithmetic.get_backend",
            return_value="numpy",
        ) as selection:
            result = execute_add(
                value, 3.0, dtype=ts.float64, output_shape=(32,),
            )
        selection.assert_called_once_with()
        self.assertIsInstance(result, NumPyStorage)
        self.assertEqual(result.buffer.tolist(), [5.0] * 32)

    def test_small_numpy_work_does_not_load_provider(self):
        value = ts.Tensor([2.0])
        with ts.use_backend("numpy"), patch(
            "tensors.backend.dispatch.arithmetic._load_array_backend",
        ) as loader:
            result = execute_add(
                value, 3.0, dtype=ts.float64, output_shape=(1,),
            )
        self.assertIsNone(result)
        loader.assert_not_called()

    def test_nested_context_restores_arithmetic_provider(self):
        value = ts.Tensor([2.0] * 32)
        with ts.use_backend("numpy"):
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
            with ts.use_backend("python"):
                self.assertNotIsInstance((value + 1.0)._storage, NumPyStorage)
            self.assertIsInstance((value + 1.0)._storage, NumPyStorage)
