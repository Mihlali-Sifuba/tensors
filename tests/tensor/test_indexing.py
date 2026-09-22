import unittest
from unittest.mock import patch

import tensors as ts
from tensors.utils.indexing import tensor_indices_to_storage_index
from tensors.utils.slicing import slice_ranges_and_shape_from_key


class TensorIndexingTests(unittest.TestCase):
    def test_reverse_slice_uses_python_semantics(self):
        tensor = ts.Tensor([1, 2, 3, 4])

        self.assertEqual(tensor[::-1].tolist(), [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(tensor[3:0:-2].tolist(), [4.0, 2.0])

    def test_single_index_selects_first_dimension(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        self.assertEqual(tensor[0].shape, (2,))
        self.assertEqual(tensor[0].tolist(), [1.0, 2.0])

    def test_tuple_indices_return_scalar(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor[1, 2], 6.0)
        self.assertEqual(tensor[-1, -2], 5.0)

    def test_complete_integer_read_uses_index_normalization_once(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        with patch(
            "tensors.tensor.tensor_indices_to_storage_index",
            wraps=tensor_indices_to_storage_index,
        ) as normalize:
            result = tensor[-1, -1]

        self.assertEqual(result, 4.0)
        normalize.assert_called_once()

    def test_slice_read_uses_slice_normalization_once(self):
        tensor = ts.Tensor([[1, 2], [3, 4], [5, 6]])

        with patch(
            "tensors.tensor.slice_ranges_and_shape_from_key",
            wraps=slice_ranges_and_shape_from_key,
        ) as normalize:
            result = tensor[1:, 1]

        self.assertEqual(result.tolist(), [4.0, 6.0])
        normalize.assert_called_once()

    def test_tuple_slice_selects_column(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        column = tensor[:, 1]

        self.assertEqual(column.shape, (2,))
        self.assertEqual(column.tolist(), [2.0, 5.0])

    def test_tuple_slice_selects_row_tail(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        row_tail = tensor[1, 1:]

        self.assertEqual(row_tail.shape, (2,))
        self.assertEqual(row_tail.tolist(), [5.0, 6.0])

    def test_partial_tuple_slice_keeps_remaining_dimensions(self):
        tensor = ts.Tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])

        result = tensor[1:]

        self.assertEqual(result.shape, (1, 2, 2))
        self.assertEqual(result.tolist(), [5.0, 6.0, 7.0, 8.0])

    def test_3d_tuple_slice_with_mixed_indices(self):
        tensor = ts.Tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ])

        result = tensor[:, 1, :]

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [3.0, 4.0, 7.0, 8.0])

    def test_empty_slice_returns_empty_tensor(self):
        result = ts.Tensor([1, 2, 3])[1:1]

        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.tolist(), [])

    def test_reverse_slice_preserves_dtype(self):
        tensor = ts.Tensor([1, 2, 3], dtype=ts.float32)

        result = tensor[::-1]

        self.assertIs(result.dtype, ts.float32)
        self.assertEqual(result.tolist(), [3.0, 2.0, 1.0])

    def test_nd_slice_preserves_dtype(self):
        tensor = ts.Tensor([[1, 2], [3, 4]], dtype=ts.int32)

        result = tensor[:, 1]

        self.assertIs(result.dtype, ts.int32)
        self.assertEqual(result.tolist(), [2, 4])

    def test_too_many_indices_are_rejected(self):
        with self.assertRaisesRegex(IndexError, "Too many indices"):
            _ = ts.Tensor([[1, 2]])[0, 0, 0]

    def test_out_of_range_index_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "Index out of range"):
            _ = ts.Tensor([1, 2])[2]

    def test_out_of_range_negative_index_is_rejected(self):
        with self.assertRaisesRegex(IndexError, "Index out of range"):
            _ = ts.Tensor([1, 2])[-3]

    def test_unsupported_index_type_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unsupported index type"):
            _ = ts.Tensor([1, 2])["bad"]

    def test_setitem_updates_1d_and_nd_values(self):
        vector = ts.Tensor([1, 2, 3])
        vector[1] = 20
        self.assertEqual(vector.tolist(), [1.0, 20.0, 3.0])

        matrix = ts.Tensor([[1, 2], [3, 4]])
        matrix[1, 0] = 30
        self.assertEqual(matrix.tolist(), [1.0, 2.0, 30.0, 4.0])

    def test_setitem_supports_negative_indices(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        tensor[-1, -1] = 40

        self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 40.0])

    def test_setitem_assigns_values_to_1d_slice(self):
        tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.float32)

        tensor[1:3] = [8, 9]

        self.assertEqual(tensor.tolist(), [1.0, 8.0, 9.0, 4.0])
        self.assertIs(tensor.dtype, ts.float32)

    def test_setitem_broadcasts_scalar_to_nd_slice(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        tensor[:, 0] = 9

        self.assertEqual(tensor.tolist(), [9.0, 2.0, 3.0, 9.0, 5.0, 6.0])

    def test_setitem_assigns_values_to_mixed_nd_slice(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        tensor[0, 1:3] = [7, 8]

        self.assertEqual(tensor.tolist(), [1.0, 7.0, 8.0, 4.0, 5.0, 6.0])

    def test_setitem_supports_strided_and_negative_slices(self):
        tensor = ts.Tensor([1, 2, 3, 4])

        tensor[-4::2] = [10, 30]

        self.assertEqual(tensor.tolist(), [10.0, 2.0, 30.0, 4.0])

    def test_setitem_broadcasts_compatible_value_shape(self):
        tensor = ts.Tensor([[1, 2, 3], [4, 5, 6]])

        tensor[:, 1:] = [8, 9]

        self.assertEqual(tensor.tolist(), [1.0, 8.0, 9.0, 4.0, 8.0, 9.0])

    def test_setitem_rejects_incompatible_slice_value_shape(self):
        tensor = ts.Tensor([1, 2, 3, 4])

        with self.assertRaisesRegex(ValueError, "Cannot assign shape"):
            tensor[1:3] = [7, 8, 9]

        self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_setitem_rejects_out_of_range_integer_in_slice_key(self):
        tensor = ts.Tensor([[1, 2], [3, 4]])

        with self.assertRaisesRegex(IndexError, "Index out of range"):
            tensor[2, :] = 0

    def test_setitem_rejects_single_index_for_nd_tensor(self):
        with self.assertRaisesRegex(ValueError, "Cannot assign"):
            ts.Tensor([[1, 2]])[0] = 99

    def test_setitem_rejects_unsupported_index_type(self):
        with self.assertRaisesRegex(TypeError, "Unsupported index type"):
            ts.Tensor([1, 2])["bad"] = 99

    def test_setitem_rejects_wrong_number_of_tuple_indices(self):
        with self.assertRaisesRegex(IndexError, "Expected 2 indices"):
            ts.Tensor([[1, 2]])[0, 0, 0] = 99

    def test_setitem_rejects_tuple_indices_for_1d_tensor(self):
        with self.assertRaisesRegex(IndexError, "Expected 1 indices"):
            ts.Tensor([1, 2])[0, 0] = 99


class IndexingExecutionTests(unittest.TestCase):
    """Where a selection runs, now that its size no longer decides.

    Splitting a stacked gradient back to the contributions it came from is a
    slice, so a reverse pass runs one, and a small selection that came back
    in Python storage left the accumulated gradient residing off the selected
    backend. Selecting elements copies them; it calculates nothing, so the
    only questions are which elements and where.
    """

    BACKENDS = ("python", "numpy", "cuda")

    #: Every selection below produces fewer elements than the workload policy
    #: this path used to apply, so each one took the Python reference before.
    SMALL = 4

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def assertSelected(self, produced, *, values, shape, backend):
        self.assertEqual(produced.tolist(), values)
        self.assertEqual(tuple(produced.shape), shape)
        self.assertIs(produced.dtype, ts.float64)
        self.assertEqual(produced.backend_storage.kind, backend)

    def test_small_selections_stay_on_the_selected_backend(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    tensor = ts.Tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64
                    )
                    row = tensor[1]
                    column = tensor[:, 2]
                    span = tensor[0, 0:2]
                    strided = tensor[0, ::2]
                    reversed_row = tensor[0, ::-1]
                self.assertSelected(
                    row, values=[4.0, 5.0, 6.0], shape=(3,), backend=backend
                )
                self.assertSelected(
                    column, values=[3.0, 6.0], shape=(2,), backend=backend
                )
                self.assertSelected(
                    span, values=[1.0, 2.0], shape=(2,), backend=backend
                )
                self.assertSelected(
                    strided, values=[1.0, 3.0], shape=(2,), backend=backend
                )
                self.assertSelected(
                    reversed_row, values=[3.0, 2.0, 1.0], shape=(3,), backend=backend
                )

    def test_a_non_contiguous_source_is_selected_in_logical_order(self):
        """A selection out of a transposed view addresses logical positions.

        The source is sized past ``execute_transpose``'s workload policy,
        which still answers small permutations from the Python reference; a
        smaller view would arrive here residing elsewhere, which is that
        dispatcher's remaining fallback rather than anything about selecting.
        """
        rows = [[float(10 * row + column) for column in range(8)] for row in range(4)]
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    transposed = ts.transpose(ts.Tensor(rows, dtype=ts.float64))
                    self.assertEqual(transposed.backend_storage.kind, backend)
                    selected = transposed[2]
                self.assertSelected(
                    selected,
                    values=[2.0, 12.0, 22.0, 32.0],
                    shape=(4,),
                    backend=backend,
                )

    def test_a_backend_that_cannot_select_a_dtype_says_so(self):
        """CUDA keeps integers off the device, and now reports it."""
        from tensors.backend.config import BackendOperationUnsupportedError

        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.int64)
                    if backend == "cuda":
                        with self.assertRaises(
                            BackendOperationUnsupportedError
                        ) as raised:
                            tensor[1:3]
                        self.assertIn("cuda", str(raised.exception))
                        self.assertIn("slice_tensor", str(raised.exception))
                        return
                    produced = tensor[1:3]
                self.assertEqual(produced.tolist(), [2, 3])
                self.assertIs(produced.dtype, ts.int64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.manipulation import slice as module

        source = inspect.getsource(module.execute_slice)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "python.kernels",
            "as reference",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("run_on_selected_backend", source)


if __name__ == "__main__":
    unittest.main()
