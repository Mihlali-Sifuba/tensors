"""In-place assignment changes a tensor's values, never its backend.

`docs/backend-storage-architecture.md` section 5.4: an in-place write updates
the tensor's own storage, on its own backend. A write whose value comes from
the host transfers that one value to the tensor's backend, which is the
host-facing write the same section allows; the tensor itself is never
transferred to the host.

Assignment previously obtained host storage and installed it as authoritative,
so writing a single element relocated a NumPy- or CUDA-resident tensor to
``PythonStorage`` permanently. These cases pin the residency that replaced
that, alongside the assignment semantics it has to leave alone: a write that
preserves the backend but changes what it writes has fixed nothing.
"""

import unittest

import tensors as ts
from tensors.backend.python.storage import PythonStorage
from tests.backend._support import BackendTestCase

BACKENDS = ("python", "numpy", "cuda")


class AssignmentResidencyTestCase(BackendTestCase):
    """Run one body under each available backend, reporting which one failed."""

    def for_each_backend(self, body):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    self.skipTest(f"the {backend} backend is not available here")
                with ts.use_backend(backend):
                    body(backend)

    def assertResident(self, tensor, backend):
        self.assertEqual(tensor.backend_storage.kind, backend)


class SingleElementAssignmentTests(AssignmentResidencyTestCase):
    def test_a_one_dimensional_element_write_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            tensor[1] = 9.0
            self.assertEqual(tensor.tolist(), [1.0, 9.0, 3.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_multidimensional_integer_index_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64)
            tensor[1, 2] = -7.0
            tensor[0, 0] = 0.5
            self.assertEqual(tensor.tolist(), [0.5, 2.0, 3.0, 4.0, 5.0, -7.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_negative_integer_index_addresses_from_the_end(self):
        def body(backend):
            tensor = ts.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ts.float64)
            tensor[-1, -1] = 8.0
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 8.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_three_dimensional_write_addresses_the_right_element(self):
        def body(backend):
            tensor = ts.Tensor(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                dtype=ts.float64,
            )
            tensor[1, 0, 1] = 60.0
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 60.0, 7.0, 8.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)


class SliceAssignmentTests(AssignmentResidencyTestCase):
    def test_a_contiguous_slice_write_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            tensor[1:3] = 0.0
            self.assertEqual(tensor.tolist(), [1.0, 0.0, 0.0, 4.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_strided_slice_write_reaches_only_its_own_positions(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=ts.float64)
            tensor[::2] = -1.0
            self.assertEqual(tensor.tolist(), [-1.0, 2.0, -1.0, 4.0, -1.0, 6.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_negative_step_slice_writes_in_reverse_order(self):
        """The step decides which value lands where, not merely which are hit."""

        def body(backend):
            tensor = ts.Tensor([0.0, 0.0, 0.0, 0.0], dtype=ts.float64)
            tensor[::-1] = [1.0, 2.0, 3.0, 4.0]
            self.assertEqual(tensor.tolist(), [4.0, 3.0, 2.0, 1.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_mixed_integer_and_slice_key_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64)
            tensor[1, 0:2] = [40.0, 50.0]
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 40.0, 50.0, 6.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_column_write_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64)
            tensor[:, 1] = 0.0
            self.assertEqual(tensor.tolist(), [1.0, 0.0, 3.0, 4.0, 0.0, 6.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_an_empty_selection_writes_nothing_and_still_counts(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            tensor[2:2] = 5.0
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(tensor.version, 1)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)


class AssignmentValueKindTests(AssignmentResidencyTestCase):
    """A scalar, a list and a Tensor are three different value origins."""

    def test_a_host_scalar_transfers_to_the_tensor_rather_than_the_reverse(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            tensor[0:2] = 7.5
            self.assertEqual(tensor.tolist(), [7.5, 7.5, 3.0, 4.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_host_list_transfers_to_the_tensor_rather_than_the_reverse(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            tensor[0:2] = [8.0, 9.0]
            self.assertEqual(tensor.tolist(), [8.0, 9.0, 3.0, 4.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_tensor_value_is_written_without_leaving_the_backend(self):
        """Both sides of the write stay put: the destination and the values."""

        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            values = ts.Tensor([8.0, 9.0], dtype=ts.float64)
            self.assertResident(values, backend)
            tensor[0:2] = values
            self.assertEqual(tensor.tolist(), [8.0, 9.0, 3.0, 4.0])
            self.assertResident(tensor, backend)
            self.assertResident(values, backend)

        self.for_each_backend(body)

    def test_the_value_tensor_is_not_aliased_by_the_destination(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            values = ts.Tensor([8.0, 9.0], dtype=ts.float64)
            tensor[0:2] = values
            values[0] = 100.0
            self.assertEqual(tensor.tolist(), [8.0, 9.0, 3.0, 4.0])
            tensor[0] = -5.0
            self.assertEqual(values.tolist(), [100.0, 9.0])

        self.for_each_backend(body)

    def test_a_two_dimensional_tensor_value_fills_a_two_dimensional_selection(self):
        def body(backend):
            tensor = ts.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=ts.float64)
            values = ts.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ts.float64)
            tensor[:, 0:2] = values
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 0.0, 3.0, 4.0, 0.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)


class AssignmentBroadcastingTests(AssignmentResidencyTestCase):
    def test_a_single_element_tensor_fills_the_whole_selection(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            tensor[0:3] = ts.Tensor([9.0], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [9.0, 9.0, 9.0, 4.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_row_is_stretched_down_a_two_dimensional_selection(self):
        """A stretched axis repeats source positions; it does not compute."""

        def body(backend):
            tensor = ts.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=ts.float64)
            tensor[0:2, :] = ts.Tensor([[1.0, 2.0, 3.0]], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_column_is_stretched_across_a_two_dimensional_selection(self):
        def body(backend):
            tensor = ts.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=ts.float64)
            tensor[0:2, :] = ts.Tensor([[1.0], [2.0]], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_shorter_rank_is_aligned_from_the_right(self):
        def body(backend):
            tensor = ts.Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=ts.float64)
            tensor[0:2, :] = [1.0, 2.0, 3.0]
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_an_incompatible_shape_is_refused_and_writes_nothing(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "Cannot assign shape"):
                tensor[0:2] = [1.0, 2.0, 3.0]
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(tensor.version, 0)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_higher_rank_value_is_refused_and_writes_nothing(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "Cannot assign shape"):
                tensor[0:2] = ts.Tensor([[1.0, 2.0]], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(tensor.version, 0)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)


class NonContiguousAssignmentTests(AssignmentResidencyTestCase):
    """A tensor whose layout does not match its storage still writes in place.

    Physical positions come from the shape, the strides and the offset, so a
    write to a transposed, reversed or offset view must reach exactly the
    storage the view addresses and leave the rest of the buffer alone.
    """

    def _synthetic(self, values, *, shape, strides, offset=0):
        storage = PythonStorage.from_values(values, ts.float64)
        return ts.Tensor._from_metadata(
            storage,
            shape=ts.Shape(*shape),
            strides=ts.Strides(*strides),
            offset=offset,
        )

    def test_a_transposed_layout_writes_the_transposed_position(self):
        tensor = self._synthetic(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(3, 2), strides=(1, 3)
        )
        self.assertFalse(tensor.is_contiguous)
        tensor[2, 0] = 30.0

        self.assertEqual(tensor.tolist(), [1.0, 4.0, 2.0, 5.0, 30.0, 6.0])
        self.assertEqual(tensor.version, 1)

    def test_a_nonzero_offset_writes_inside_the_buffer_it_addresses(self):
        tensor = self._synthetic(
            [99.0, 1.0, 2.0, 3.0, 88.0], shape=(3,), strides=(1,), offset=1
        )
        tensor[0] = 10.0
        tensor[2] = 30.0

        self.assertEqual(tensor.tolist(), [10.0, 2.0, 30.0])
        self.assertEqual(tensor.backend_storage.buffer[0], 99.0)
        self.assertEqual(tensor.backend_storage.buffer[4], 88.0)

    def test_a_negative_stride_view_writes_in_its_own_order(self):
        tensor = self._synthetic(
            [1.0, 2.0, 3.0, 4.0], shape=(4,), strides=(-1,), offset=3
        )
        self.assertEqual(tensor.tolist(), [4.0, 3.0, 2.0, 1.0])
        tensor[0:2] = [40.0, 30.0]

        self.assertEqual(tensor.tolist(), [40.0, 30.0, 2.0, 1.0])
        self.assertEqual(tensor.backend_storage.buffer.tolist(), [1.0, 2.0, 30.0, 40.0])

    def test_a_strided_slice_of_an_offset_view_writes_only_its_positions(self):
        tensor = self._synthetic(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=(3,), strides=(2,), offset=1
        )
        self.assertEqual(tensor.tolist(), [1.0, 3.0, 5.0])
        tensor[::2] = -1.0

        self.assertEqual(tensor.tolist(), [-1.0, 3.0, -1.0])
        self.assertEqual(
            tensor.backend_storage.buffer.tolist(),
            [0.0, -1.0, 2.0, 3.0, 4.0, -1.0, 6.0],
        )


class AssignmentConversionTests(AssignmentResidencyTestCase):
    """Assignment casts as construction casts, and refuses what it refuses."""

    def test_an_integer_tensor_truncates_a_float_toward_zero(self):
        def body(backend):
            tensor = ts.Tensor([1, 2, 3, 4], dtype=ts.int32)
            tensor[0] = 5.7
            tensor[1:3] = -2.9
            self.assertEqual(tensor.tolist(), [5, -2, -2, 4])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_float_tensor_accepts_an_integer_value(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            tensor[0] = 3
            self.assertEqual(tensor.tolist(), [3.0, 2.0])
            self.assertIs(tensor.dtype, ts.float64)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_value_tensor_of_another_dtype_is_converted_to_the_destination(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            tensor[0:2] = ts.Tensor([8.0, 9.0], dtype=ts.float32)
            self.assertEqual(tensor.tolist(), [8.0, 9.0, 3.0])
            self.assertIs(tensor.dtype, ts.float64)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_non_numeric_value_is_refused(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            with self.assertRaisesRegex(TypeError, "must be numeric"):
                tensor[0] = "x"
            with self.assertRaisesRegex(TypeError, "must be a number, list"):
                tensor[0:1] = "x"
            self.assertEqual(tensor.tolist(), [1.0, 2.0])
            self.assertEqual(tensor.version, 0)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_an_integer_too_wide_for_the_dtype_is_refused(self):
        def body(backend):
            tensor = ts.Tensor([1, 2, 3], dtype=ts.int32)
            with self.assertRaises(OverflowError):
                tensor[0] = 2**40
            with self.assertRaises(OverflowError):
                tensor[0:2] = 2**40
            self.assertEqual(tensor.tolist(), [1, 2, 3])
            self.assertEqual(tensor.version, 0)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_an_item_assignment_requires_exactly_one_value(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "exactly one value"):
                tensor[0] = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [1.0, 2.0])
            self.assertResident(tensor, backend)

        self.for_each_backend(body)


class AssignmentStorageStateTests(AssignmentResidencyTestCase):
    """Storage identity, cache invalidation and mutation versioning."""

    def test_the_authoritative_storage_object_is_written_in_place(self):
        """The tensor keeps the storage it had; only the values differ."""

        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            before = tensor.backend_storage
            tensor[0] = 5.0
            tensor[1:3] = 6.0
            self.assertIs(tensor.backend_storage, before)
            self.assertEqual(tensor.tolist(), [5.0, 6.0, 6.0])

        self.for_each_backend(body)

    def test_a_representation_converted_before_the_write_is_dropped(self):
        """A host read populates a cache entry the write then makes stale."""

        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0])
            tensor[0] = 42.0
            self.assertEqual(
                set(tensor.backend_storage_cache), {tensor.backend_storage.kind}
            )
            self.assertIs(tensor.backend_storage_cache[backend], tensor.backend_storage)
            self.assertEqual(tensor.tolist(), [42.0, 2.0, 3.0])

        self.for_each_backend(body)

    def test_repeated_writes_each_advance_the_version(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            self.assertEqual(tensor.version, 0)
            tensor[0] = 1.5
            self.assertEqual(tensor.version, 1)
            tensor[0:2] = 2.5
            self.assertEqual(tensor.version, 2)
            tensor[1,] = 3.5
            self.assertEqual(tensor.version, 3)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_failed_write_changes_neither_values_version_nor_backend(self):
        def body(backend):
            tensor = ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64)
            before = tensor.backend_storage
            with self.assertRaises(IndexError):
                tensor[9] = 1.0
            self.assertEqual(tensor.tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(tensor.version, 0)
            self.assertIs(tensor.backend_storage, before)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_slice_write_rejected_partway_through_its_values_is_atomic(self):
        def body(backend):
            tensor = ts.Tensor([1, 2, 3], dtype=ts.int32)
            with self.assertRaises(TypeError):
                tensor[0:2] = [4, object()]
            self.assertEqual(tensor.tolist(), [1, 2, 3])
            self.assertEqual(tensor.version, 0)
            self.assertResident(tensor, backend)

        self.for_each_backend(body)

    def test_a_clone_is_written_independently_of_its_source(self):
        def body(backend):
            original = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            clone = original.clone()
            clone[0] = 9.0
            self.assertEqual(original.tolist(), [1.0, 2.0])
            self.assertEqual(clone.tolist(), [9.0, 2.0])
            self.assertEqual(original.version, 0)
            self.assertEqual(clone.version, 1)
            self.assertResident(original, backend)
            self.assertResident(clone, backend)

        self.for_each_backend(body)


class LargeAssignmentTests(AssignmentResidencyTestCase):
    """Residency does not depend on how much is written.

    A workload-size threshold deciding where an operation runs is exactly what
    made a small gradient come back in host storage elsewhere in the package.
    A write has no such threshold, so the same cases hold at both sizes.
    """

    def test_a_large_scalar_fill_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([0.0] * 50_000, dtype=ts.float64)
            tensor[0:50_000] = 3.0
            self.assertResident(tensor, backend)
            self.assertEqual(tensor[0], 3.0)
            self.assertEqual(tensor[49_999], 3.0)
            self.assertEqual(tensor.version, 1)

        self.for_each_backend(body)

    def test_a_large_tensor_valued_write_keeps_the_backend(self):
        def body(backend):
            tensor = ts.Tensor([0.0] * 20_000, dtype=ts.float64)
            values = ts.Tensor([1.0] * 10_000, dtype=ts.float64)
            tensor[0:10_000] = values
            self.assertResident(tensor, backend)
            self.assertEqual(tensor[0], 1.0)
            self.assertEqual(tensor[9_999], 1.0)
            self.assertEqual(tensor[10_000], 0.0)

        self.for_each_backend(body)

    def test_a_large_strided_write_touches_only_its_own_positions(self):
        def body(backend):
            tensor = ts.Tensor([0.0] * 20_000, dtype=ts.float64)
            tensor[::2] = 5.0
            self.assertResident(tensor, backend)
            self.assertEqual(tensor[0], 5.0)
            self.assertEqual(tensor[1], 0.0)
            self.assertEqual(tensor[19_998], 5.0)
            self.assertEqual(tensor[19_999], 0.0)

        self.for_each_backend(body)

    def test_a_small_and_a_large_write_agree_element_for_element(self):
        """Nothing about the result may depend on the amount written."""

        def body(backend):
            for size in (4, 4_096):
                small = ts.Tensor([float(index) for index in range(size)])
                small[1::2] = -1.0
                expected = [
                    -1.0 if index % 2 else float(index) for index in range(size)
                ]
                self.assertEqual(small.tolist(), expected)
                self.assertResident(small, backend)

        self.for_each_backend(body)


class MutationAndDifferentiationTests(AssignmentResidencyTestCase):
    """Writing to a recorded tensor is still detected after the fix."""

    def test_a_write_to_a_recorded_variable_is_visible_to_the_graph(self):
        def body(backend):
            from tensors.graph.state import reset_graph_state

            reset_graph_state()
            value = ts.Tensor([2.0, 3.0], dtype=ts.float64)
            variable = ts.Variable(value, requires_grad=True)
            before = variable._mutation_state()
            value[0] = 5.0
            self.assertNotEqual(variable._mutation_state(), before)
            self.assertEqual(value.version, 1)
            self.assertResident(value, backend)
            reset_graph_state()

        self.for_each_backend(body)

    def test_differentiation_after_a_write_uses_the_written_values(self):
        def body(backend):
            from tensors.graph.state import reset_graph_state

            reset_graph_state()
            value = ts.Tensor([2.0, 3.0], dtype=ts.float64)
            value[0] = 4.0
            variable = ts.Variable(value, requires_grad=True)
            gradient = ts.grad(
                variable * variable,
                variable,
                grad_outputs=ts.Tensor([1.0, 1.0], dtype=ts.float64),
            )
            self.assertEqual(gradient.tolist(), [8.0, 6.0])
            self.assertResident(gradient, backend)
            reset_graph_state()

        self.for_each_backend(body)


class GradcheckIntegrationTests(AssignmentResidencyTestCase):
    """``gradcheck`` perturbs its inputs by assignment, so it needed this fix.

    Under strict selection the perturbed clone used to arrive at the next
    operation in host storage, and the residency check rejected it before any
    derivative was compared.
    """

    def test_gradcheck_verifies_a_product_under_each_backend(self):
        def body(backend):
            inputs = ts.Tensor([1.5, -2.0, 0.5], dtype=ts.float64)
            self.assertTrue(ts.gradcheck(lambda value: value * value, inputs))

        self.for_each_backend(body)

    def test_gradcheck_verifies_a_two_input_expression_under_each_backend(self):
        def body(backend):
            left = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            right = ts.Tensor([3.0, -1.0], dtype=ts.float64)
            self.assertTrue(ts.gradcheck(lambda a, b: a * b + a, (left, right)))

        self.for_each_backend(body)

    def test_gradcheck_verifies_a_multidimensional_input_under_each_backend(self):
        def body(backend):
            inputs = ts.Tensor([[1.0, 2.0], [-0.5, 3.0]], dtype=ts.float64)
            self.assertTrue(ts.gradcheck(lambda value: value * value, inputs))

        self.for_each_backend(body)

    def test_gradcheck_leaves_its_inputs_on_their_backend(self):
        def body(backend):
            inputs = ts.Tensor([1.0, 2.0], dtype=ts.float64)
            ts.gradcheck(lambda value: value * value, inputs)
            self.assertResident(inputs, backend)
            self.assertEqual(inputs.version, 0)

        self.for_each_backend(body)


if __name__ == "__main__":
    unittest.main()
