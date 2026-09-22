import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state


class StackTests(unittest.TestCase):
    def test_stack(self):
        left = ts.Tensor([1, 2, 3])
        right = ts.Tensor([4, 5, 6])

        stacked_rows = ts.stack([left, right])
        self.assertEqual(stacked_rows.shape, (2, 3))
        self.assertEqual(stacked_rows.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        stacked_columns = ts.stack([left, right], axis=1)
        self.assertEqual(stacked_columns.shape, (3, 2))
        self.assertEqual(stacked_columns.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_stack_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            ts.stack([ts.Tensor([1, 2]), ts.Tensor([1, 2, 3])])

    def test_stack_with_lists(self):
        stacked = ts.stack([[1, 2], [3, 4]])

        self.assertEqual(stacked.shape, (2, 2))
        self.assertEqual(stacked.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stack_negative_axis(self):
        tensor = ts.Tensor([1, 2])
        stacked = ts.stack([tensor, tensor], axis=-1)

        self.assertEqual(stacked.shape, (2, 2))
        self.assertEqual(stacked.tolist(), [1.0, 1.0, 2.0, 2.0])

    def test_stack_rejects_empty_sequence(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ts.stack([])

    def test_stack_rejects_axis_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            ts.stack([ts.Tensor([1, 2])], axis=3)

    def test_stack_rejects_non_integer_axes(self):
        tensors = [ts.Tensor([1]), ts.Tensor([2])]

        with self.assertRaisesRegex(TypeError, "integer"):
            ts.stack(tensors, axis=False)
        with self.assertRaisesRegex(TypeError, "integer"):
            ts.stack(tensors, axis=0.0)

    def test_stack_preserves_first_tensor_dtype(self):
        left = ts.Tensor([1, 2], dtype=ts.float32)
        right = ts.Tensor([3, 4], dtype=ts.float32)

        result = ts.stack([left, right])

        self.assertIs(result.dtype, ts.float32)

    def test_stack_restores_each_input_gradient_dtype(self):
        left = ts.Variable(ts.Tensor([1.0], dtype=ts.float32))
        right = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))

        ts.backward(ts.sum(ts.stack([left, right])))

        self.assertIs(left.grad.dtype, ts.float32)
        self.assertIs(right.grad.dtype, ts.float64)
        self.assertEqual(left.grad.tolist(), [1.0])
        self.assertEqual(right.grad.tolist(), [1.0])

    def test_math_namespace_exposes_stack_operation_class(self):
        result = ts.math.Stack().forward([ts.Tensor([1, 2]), ts.Tensor([3, 4])])

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_stack_variables_is_differentiable(self):
        left = ts.Variable([1.0, 2.0])
        right = ts.Variable([3.0, 4.0])

        loss = ts.sum(ts.stack([left, right], axis=1) ** 2.0)
        ts.backward(loss)

        self.assertEqual(left.grad.tolist(), [2.0, 4.0])
        self.assertEqual(right.grad.tolist(), [6.0, 8.0])


class StackExecutionTests(unittest.TestCase):
    """Where stacking runs, now that its size no longer decides.

    Stacking arranges values; it does not calculate one. So the question each
    case asks is whether the arrangement is the specified one and whether it
    happened on the backend that was selected — at sizes far below the
    workload threshold this path used to branch on.
    """

    BACKENDS = ("python", "numpy", "cuda")

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def test_a_small_stack_stays_on_the_selected_backend(self):
        """Four elements: well under the policy this path used to apply."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    produced = ts.stack(
                        [
                            ts.Tensor([1.0, 2.0], dtype=ts.float64),
                            ts.Tensor([3.0, 4.0], dtype=ts.float64),
                        ]
                    )
                self.assertEqual(produced.tolist(), [1.0, 2.0, 3.0, 4.0])
                self.assertEqual(tuple(produced.shape), (2, 2))
                self.assertIs(produced.dtype, ts.float64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_axis_handling_is_unchanged_on_every_backend(self):
        """Axis 0 interleaves rows; axis 1 and -1 interleave columns."""
        expectations = {
            0: ([1.0, 2.0, 3.0, 4.0], (2, 2)),
            1: ([1.0, 3.0, 2.0, 4.0], (2, 2)),
            -1: ([1.0, 3.0, 2.0, 4.0], (2, 2)),
        }
        for backend in self.BACKENDS:
            for axis, (values, shape) in expectations.items():
                with self.subTest(backend=backend, axis=axis):
                    self._require(backend)
                    with ts.use_backend(backend):
                        produced = ts.stack(
                            [
                                ts.Tensor([1.0, 2.0], dtype=ts.float64),
                                ts.Tensor([3.0, 4.0], dtype=ts.float64),
                            ],
                            axis=axis,
                        )
                    self.assertEqual(produced.tolist(), values)
                    self.assertEqual(tuple(produced.shape), shape)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_non_contiguous_input_is_read_in_logical_order(self):
        """A transposed view stacks by what it addresses, not by its buffer.

        The source is deliberately larger than ``transpose`` needs to stay on
        the selected backend. ``execute_transpose`` still carries a
        workload-size policy, so a smaller view would arrive here residing in
        Python and be rejected — which is that dispatcher's remaining
        fallback, not something about stacking.
        """
        columns = 8
        rows = [[float(column) for column in range(columns)] for _ in range(4)]
        expected_transpose = [
            float(column) for column in range(columns) for _ in range(4)
        ]
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    transposed = ts.transpose(ts.Tensor(rows, dtype=ts.float64))
                    self.assertEqual(tuple(transposed.shape), (columns, 4))
                    self.assertEqual(transposed.backend_storage.kind, backend)
                    produced = ts.stack([transposed, transposed], axis=0)
                self.assertEqual(tuple(produced.shape), (2, columns, 4))
                # Each copy is the transpose read row by row.
                self.assertEqual(produced.tolist(), expected_transpose * 2)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_foreign_resident_operand_is_rejected_before_the_kernel(self):
        """The mismatch is reported, not quietly converted on the way in.

        The operands reach the boundary inside one sequence, which the
        residency check reads as a single value carrying no residency. Were
        they not checked separately, the kernel would lower them and the
        stack would silently move backends.
        """
        if "numpy" not in ts.available_backends():
            self.skipTest("the numpy backend is not available here")
        import tensors.backend.numpy.kernels as numpy_kernels

        with ts.use_backend("python"):
            foreign = ts.Tensor([1.0, 2.0], dtype=ts.float64)
        with ts.use_backend("numpy"):
            resident = ts.Tensor([3.0, 4.0], dtype=ts.float64)

            original = numpy_kernels.stack
            with patch.object(numpy_kernels, "stack", wraps=original) as kernel:
                backend_state._clear_backend_kernel_cache()
                with self.assertRaises(ts.BackendMismatchError) as raised:
                    ts.stack([foreign, resident])
                backend_state._clear_backend_kernel_cache()
                self.assertEqual(kernel.call_count, 0, "rejected before the kernel")

                # The position of the offending operand is what is reported.
                self.assertIn("python", str(raised.exception))
                self.assertIn("numpy", str(raised.exception))

                # A resident pair goes through and reaches that same kernel.
                backend_state._clear_backend_kernel_cache()
                produced = ts.stack([resident, resident])
                backend_state._clear_backend_kernel_cache()
            self.assertEqual(kernel.call_count, 1)
        self.assertEqual(produced.tolist(), [3.0, 4.0, 3.0, 4.0])
        self.assertEqual(produced.backend_storage.kind, "numpy")

    def test_a_backend_that_cannot_stack_a_dtype_says_so(self):
        """CUDA keeps integers off the device, and now reports it.

        Its result conversion declines an integer dtype, which used to send
        the work to the Python reference. Removing that fallback makes the
        decline visible instead of silently relocating the computation.
        """
        from tensors.backend.config import BackendOperationUnsupportedError

        integers = ([1, 2, 3], [4, 5, 6])
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    operands = [ts.Tensor(v, dtype=ts.int64) for v in integers]
                    if backend == "cuda":
                        with self.assertRaises(
                            BackendOperationUnsupportedError
                        ) as raised:
                            ts.stack(operands)
                        message = str(raised.exception)
                        self.assertIn("cuda", message)
                        self.assertIn("stack", message)
                        return
                    produced = ts.stack(operands)
                self.assertEqual(produced.tolist(), [1, 2, 3, 4, 5, 6])
                self.assertIs(produced.dtype, ts.int64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_the_dispatcher_carries_no_threshold_or_fallback(self):
        import inspect

        from tensors.backend.dispatch.manipulation import stack as module

        source = inspect.getsource(module.execute_stack)
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
