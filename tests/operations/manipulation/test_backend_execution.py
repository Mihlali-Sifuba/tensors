"""Strict selected-backend execution for remaining manipulation operations."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
import tensors.tensor as tensor_module
from tensors.backend.config import (
    BackendMismatchError,
    BackendOperationUnsupportedError,
)
from tensors.graph.state import reset_graph_state
from tensors.operations.manipulation.slice import SliceScatter


class HostReadCounter:
    """Count accesses to the Tensor host-value compatibility property."""

    def __init__(self):
        self.reads = 0
        self._original = tensor_module.Tensor._data.fget

    def __enter__(self):
        counter = self

        def counting(instance):
            counter.reads += 1
            return counter._original(instance)

        self._patch = patch.object(
            tensor_module.Tensor,
            "_data",
            property(counting),
        )
        self._patch.__enter__()
        return self

    def __exit__(self, *arguments):
        return self._patch.__exit__(*arguments)


class ManipulationBackendExecutionTests(unittest.TestCase):
    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    continue
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def test_small_and_large_concat_stay_on_the_selected_backend(self):
        def body(backend):
            scalar = ts.concat(
                [
                    ts.Tensor([1.0], shape=(), dtype=ts.float32),
                    ts.Tensor([2.0], shape=(), dtype=ts.float32),
                ]
            )
            small = ts.concat(
                [
                    ts.Tensor([[1.0], [2.0]], dtype=ts.float64),
                    ts.Tensor([[3.0, 4.0], [5.0, 6.0]], dtype=ts.float64),
                ],
                axis=1,
            )
            large = ts.concat(
                [ts.full((64, 16), 2.0), ts.full((64, 8), 3.0)],
                axis=-1,
            )
            self.assertEqual(scalar.tolist(), [1.0, 2.0])
            self.assertEqual(scalar.shape, (2,))
            self.assertIs(scalar.dtype, ts.float32)
            self.assertEqual(small.tolist(), [1.0, 3.0, 4.0, 2.0, 5.0, 6.0])
            self.assertEqual(small.shape, (2, 3))
            self.assertEqual(large.shape, (64, 24))
            self.assertEqual(large.tolist()[0], 2.0)
            self.assertEqual(large.tolist()[-1], 3.0)
            for result in (scalar, small, large):
                self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_concat_preserves_promotion_empty_axes_and_owned_storage(self):
        def body(backend):
            left = ts.Tensor([[1], [2]], dtype=ts.int32)
            right = ts.Tensor([[3.5], [4.5]], dtype=ts.float64)
            promoted = ts.concat([left, right], axis=0)
            empty = ts.concat(
                [
                    ts.Tensor([], dtype=ts.float32, shape=(0, 2)),
                    ts.Tensor([[1.0, 2.0]], dtype=ts.float32),
                ],
                axis=0,
            )
            left[0, 0] = 9
            self.assertIs(promoted.dtype, ts.float64)
            self.assertEqual(promoted.tolist(), [1.0, 2.0, 3.5, 4.5])
            self.assertEqual(empty.shape, (1, 2))
            self.assertEqual(empty.tolist(), [1.0, 2.0])
            self.assertEqual(promoted.backend_storage.kind, backend)
            self.assertEqual(empty.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_transpose_permutations_sizes_dtypes_and_ownership(self):
        def body(backend):
            small_source = ts.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
            small = ts.transpose(small_source)
            permuted = ts.transpose(
                ts.Tensor(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dtype=ts.float32,
                    shape=(1, 2, 3),
                ),
                axes=(2, 0, 1),
            )
            large = ts.transpose(
                ts.reshape(ts.arange(1024, dtype=ts.float64), (32, 32))
            )
            small_source[0, 0] = 99
            self.assertEqual(small.tolist(), [1, 4, 2, 5, 3, 6])
            self.assertEqual(small.shape, (3, 2))
            self.assertIs(small.dtype, ts.int32)
            self.assertEqual(permuted.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            self.assertEqual(permuted.shape, (3, 1, 2))
            self.assertIs(permuted.dtype, ts.float32)
            self.assertEqual(large.shape, (32, 32))
            for result in (small, permuted, large):
                self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_noncompact_inputs_use_logical_order_without_host_reads(self):
        def body(backend):
            base = ts.Tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                dtype=ts.float64,
            )
            view = ts.Tensor._from_metadata(
                base.backend_storage,
                shape=(3, 2),
                strides=(1, 3),
            )
            gradient_view = ts.Tensor._from_metadata(
                base.backend_storage,
                shape=(2,),
                strides=(2,),
            )
            with HostReadCounter() as counter:
                joined = ts.concat([view, view], axis=0)
                transposed = ts.transpose(view)
                scattered = SliceScatter(
                    source_shape=(4,),
                    key=slice(1, 3),
                ).forward(gradient_view)
            self.assertEqual(counter.reads, 0)
            self.assertEqual(
                joined.tolist(),
                [1.0, 4.0, 2.0, 5.0, 3.0, 6.0] * 2,
            )
            self.assertEqual(transposed.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            self.assertEqual(scattered.tolist(), [0.0, 1.0, 3.0, 0.0])
            self.assertEqual(joined.backend_storage.kind, backend)
            self.assertEqual(transposed.backend_storage.kind, backend)
            self.assertEqual(scattered.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_slice_scatter_first_and_higher_order_stay_resident(self):
        def body(backend):
            value = ts.Variable(ts.Tensor([1.0, 2.0, 3.0]))
            loss = ts.sum(value[1:] ** 3.0)
            first = ts.grad(loss, value, create_graph=True)
            second = ts.grad(
                first,
                value,
                grad_outputs=ts.Tensor([1.0, 1.0, 1.0]),
            )
            self.assertEqual(first.data.tolist(), [0.0, 12.0, 27.0])
            self.assertEqual(second.tolist(), [0.0, 12.0, 18.0])
            self.assertEqual(first.data.backend_storage.kind, backend)
            self.assertEqual(second.backend_storage.kind, backend)

        self.for_each_backend(body)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_numpy_kernels_receive_native_values_without_python_fallback(self):
        original_concat = numpy_backend.concat
        original_scatter = numpy_backend.slice_scatter
        original_transpose = numpy_backend.transpose

        def concat_spy(values, shapes, *args, **kwargs):
            self.assertTrue(all(isinstance(value, numpy.ndarray) for value in values))
            self.assertEqual(tuple(value.shape for value in values), shapes)
            return original_concat(values, shapes, *args, **kwargs)

        def scatter_spy(value, *args, **kwargs):
            self.assertIsInstance(value, numpy.ndarray)
            return original_scatter(value, *args, **kwargs)

        def transpose_spy(value, *args, **kwargs):
            self.assertIsInstance(value, numpy.ndarray)
            return original_transpose(value, *args, **kwargs)

        with (
            patch.object(numpy_backend, "concat", side_effect=concat_spy) as concat,
            patch.object(
                numpy_backend,
                "slice_scatter",
                side_effect=scatter_spy,
            ) as scatter,
            patch.object(
                numpy_backend,
                "transpose",
                side_effect=transpose_spy,
            ) as transpose,
            patch.object(
                python_backend,
                "concat",
                side_effect=AssertionError("Python concat fallback executed"),
            ),
            patch.object(
                python_backend,
                "slice_scatter",
                side_effect=AssertionError("Python scatter fallback executed"),
            ),
            patch.object(
                python_backend,
                "transpose",
                side_effect=AssertionError("Python transpose fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            ts.concat([ts.Tensor([1.0]), ts.Tensor([2.0])])
            ts.transpose(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))
            value = ts.Variable(ts.Tensor([1.0, 2.0, 3.0]))
            ts.grad(ts.sum(value[1:]), value)
        concat.assert_called_once()
        scatter.assert_called_once()
        transpose.assert_called_once()

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_foreign_residency_is_rejected_before_native_kernels(self):
        with ts.use_backend("python"):
            vector = ts.Tensor([1.0, 2.0])
            matrix = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        with (
            patch.object(numpy_backend, "concat") as concat,
            patch.object(numpy_backend, "slice_scatter") as scatter,
            patch.object(numpy_backend, "transpose") as transpose,
            ts.use_backend("numpy"),
        ):
            with self.assertRaises(BackendMismatchError):
                ts.concat([vector, vector])
            with self.assertRaises(BackendMismatchError):
                ts.transpose(matrix)
            with self.assertRaises(BackendMismatchError):
                SliceScatter(source_shape=(2,), key=slice(None)).forward(vector)
        concat.assert_not_called()
        scatter.assert_not_called()
        transpose.assert_not_called()

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_kernel_declines_raise_without_python_fallback(self):
        with ts.use_backend("numpy"):
            with patch.object(numpy_backend, "concat", return_value=None):
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.concat([ts.Tensor([1.0]), ts.Tensor([2.0])])
            with patch.object(numpy_backend, "transpose", return_value=None):
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.transpose(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))
            with patch.object(numpy_backend, "slice_scatter", return_value=None):
                with self.assertRaises(BackendOperationUnsupportedError):
                    SliceScatter(source_shape=(2,), key=slice(None)).forward(
                        ts.Tensor([1.0, 2.0])
                    )

    def test_dispatchers_have_no_threshold_or_reference_fallback(self):
        repository = Path(__file__).resolve().parents[3]
        for filename in ("concat.py", "slice_scatter.py", "transpose.py"):
            source = (
                repository
                / "tensors"
                / "backend"
                / "dispatch"
                / "manipulation"
                / filename
            ).read_text(encoding="utf-8")
            for forbidden in (
                "_array_work_is_large_enough",
                "_NUMPY_ELEMENTWISE_MIN_SIZE",
                "_backend_kernel",
                "python.kernels",
                "as reference",
            ):
                self.assertNotIn(forbidden, source)
            self.assertIn("validate_backend_residency", source)
            self.assertIn("BackendOperationUnsupportedError", source)

    def test_accelerated_kernels_have_no_tensor_semantics_or_python_loops(self):
        repository = Path(__file__).resolve().parents[3]
        for backend in ("numpy", "cuda"):
            for filename in ("concat.py", "slice_scatter.py", "transpose.py"):
                with self.subTest(backend=backend, filename=filename):
                    source = (
                        repository
                        / "tensors"
                        / "backend"
                        / backend
                        / "kernels"
                        / "manipulation"
                        / filename
                    ).read_text(encoding="utf-8")
                    tree = ast.parse(source)
                    loops = [
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.For, ast.AsyncFor, ast.While))
                    ]
                    self.assertEqual(loops, [])
                    for forbidden in (
                        "Tensor",
                        "._data",
                        ".tolist(",
                        ".get(",
                        "asnumpy",
                        "tensor_to_logical_array",
                    ):
                        self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
