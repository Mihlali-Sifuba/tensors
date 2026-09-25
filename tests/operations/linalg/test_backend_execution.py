"""Strict selected-backend execution for linear algebra."""

import ast
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy

import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state

BACKENDS = ("python", "numpy", "cuda")


class LinearAlgebraBackendExecutionTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    continue
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def test_matmul_forms_have_concrete_values_shapes_and_residency(self):
        cases = (
            ([1.0, 2.0], (2,), [3.0, 4.0], (2,), (), [11.0]),
            (
                [1.0, 2.0, 3.0, 4.0],
                (2, 2),
                [5.0, 6.0],
                (2,),
                (2,),
                [17.0, 39.0],
            ),
            (
                [5.0, 6.0],
                (2,),
                [1.0, 2.0, 3.0, 4.0],
                (2, 2),
                (2,),
                [23.0, 34.0],
            ),
            (
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                (2, 3),
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                (3, 2),
                (2, 2),
                [58.0, 64.0, 139.0, 154.0],
            ),
        )

        def body(backend):
            for (
                left_data,
                left_shape,
                right_data,
                right_shape,
                shape,
                expected,
            ) in cases:
                with self.subTest(left_shape=left_shape, right_shape=right_shape):
                    left = ts.Tensor(left_data, shape=left_shape)
                    right = ts.Tensor(right_data, shape=right_shape)
                    result = ts.matmul(left, right)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.tolist(), expected)
                    self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_batched_broadcast_and_empty_contractions_are_native(self):
        def body(backend):
            left = ts.Tensor([1.0, 2.0], shape=(1, 1, 2))
            right = ts.Tensor([3.0, 4.0, 5.0, 6.0], shape=(2, 2, 1))
            result = ts.matmul(left, right)
            self.assertEqual(result.shape, (2, 1, 1))
            self.assertEqual(result.tolist(), [11.0, 17.0])
            empty = ts.matmul(ts.Tensor([], shape=(2, 0)), ts.Tensor([], shape=(0, 3)))
            self.assertEqual(empty.shape, (2, 3))
            self.assertEqual(empty.tolist(), [0.0] * 6)
            self.assertEqual(result.backend_storage.kind, backend)
            self.assertEqual(empty.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_workload_size_never_changes_selected_backend(self):
        def body(backend):
            for size in (1, 31, 32, 33):
                left = ts.full((size, 1), 2.0)
                right = ts.full((1, size), 3.0)
                result = ts.matmul(left, right)
                self.assertEqual(result.shape, (size, size))
                self.assertEqual(result.tolist()[0], 6.0)
                self.assertEqual(result.tolist()[-1], 6.0)
                self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_dtype_promotion_and_outer_values_are_preserved(self):
        def body(backend):
            left = ts.Tensor([[1.0, 2.0]], dtype=ts.float32)
            right = ts.Tensor([[3.0], [4.0]], dtype=ts.float64)
            product = ts.matmul(left, right)
            outer = ts.outer(ts.Tensor([1.0, 2.0]), ts.Tensor([3.0, 4.0, 5.0]))
            self.assertIs(product.dtype, ts.float64)
            self.assertEqual(product.tolist(), [11.0])
            self.assertEqual(outer.shape, (2, 3))
            self.assertEqual(outer.tolist(), [3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
            self.assertEqual(product.backend_storage.kind, backend)
            self.assertEqual(outer.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_first_order_matmul_vjps_have_concrete_values_and_residency(self):
        def body(backend):
            left = ts.Variable(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))
            right = ts.Variable(ts.Tensor([[5.0, 6.0], [7.0, 8.0]]))
            output = ts.matmul(left, right)
            left_grad, right_grad = ts.grad(
                output,
                (left, right),
                grad_outputs=ts.full(output.shape, 1.0),
            )
            self.assertEqual(left_grad.tolist(), [11.0, 15.0, 11.0, 15.0])
            self.assertEqual(right_grad.tolist(), [4.0, 4.0, 6.0, 6.0])
            self.assertEqual(left_grad.shape, left.shape)
            self.assertEqual(right_grad.shape, right.shape)
            self.assertIs(left_grad.dtype, ts.float64)
            self.assertEqual(left_grad.backend_storage.kind, backend)
            self.assertEqual(right_grad.backend_storage.kind, backend)

            vector_left = ts.Variable(ts.Tensor([1.0, 2.0]))
            vector_right = ts.Variable(ts.Tensor([3.0, 4.0]))
            scalar = ts.matmul(vector_left, vector_right)
            vector_left_grad, vector_right_grad = ts.grad(
                scalar,
                (vector_left, vector_right),
                grad_outputs=ts.Tensor(2.0),
            )
            self.assertEqual(vector_left_grad.tolist(), [6.0, 8.0])
            self.assertEqual(vector_right_grad.tolist(), [2.0, 4.0])

        self.for_each_backend(body)

    def test_broadcast_matmul_and_outer_vjps_reduce_natively(self):
        def body(backend):
            left = ts.Variable(ts.Tensor([1.0, 2.0], shape=(1, 1, 2)))
            right = ts.Variable(ts.Tensor([3.0, 4.0, 5.0, 6.0], shape=(2, 2, 1)))
            output = ts.matmul(left, right)
            left_grad, right_grad = ts.grad(
                output,
                (left, right),
                grad_outputs=ts.full(output.shape, 1.0),
            )
            self.assertEqual(left_grad.tolist(), [8.0, 10.0])
            self.assertEqual(right_grad.tolist(), [1.0, 2.0, 1.0, 2.0])
            self.assertEqual(left_grad.backend_storage.kind, backend)
            self.assertEqual(right_grad.backend_storage.kind, backend)

            outer_left = ts.Variable(ts.Tensor([1.0, 2.0]))
            outer_right = ts.Variable(ts.Tensor([3.0, 4.0, 5.0]))
            outer_output = ts.outer(outer_left, outer_right)
            outer_left_grad, outer_right_grad = ts.grad(
                outer_output,
                (outer_left, outer_right),
                grad_outputs=ts.full(outer_output.shape, 1.0),
            )
            self.assertEqual(outer_left_grad.tolist(), [12.0, 12.0])
            self.assertEqual(outer_right_grad.tolist(), [3.0, 3.0, 3.0])
            self.assertEqual(outer_left_grad.backend_storage.kind, backend)
            self.assertEqual(outer_right_grad.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_ieee_forward_cases_and_unsupported_boundaries_are_explicit(self):
        def body(backend):
            positive = ts.matmul(ts.Tensor([math.inf]), ts.Tensor([1.0])).item()
            invalid = ts.matmul(ts.Tensor([math.inf]), ts.Tensor([0.0])).item()
            zero = ts.matmul(ts.Tensor([-0.0]), ts.Tensor([1.0])).item()
            self.assertEqual(positive, math.inf)
            self.assertTrue(math.isnan(invalid))
            self.assertEqual(math.copysign(1.0, zero), 1.0)
            if backend != "python":
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.matmul(
                        ts.Tensor([1e308, 1e308, -1e308, -1e308]),
                        ts.Tensor([1.0, 1.0, 1.0, 1.0]),
                    )
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.matmul(
                        ts.Tensor([[1, 2]], dtype=ts.int32),
                        ts.Tensor([[3], [4]], dtype=ts.int32),
                    )
                with self.assertRaises(BackendOperationUnsupportedError):
                    ts.outer(
                        ts.Tensor([1, 2], dtype=ts.int32),
                        ts.Tensor([3, 4], dtype=ts.int32),
                    )

        self.for_each_backend(body)

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_numpy_kernels_receive_native_values_without_python_fallback(self):
        original_matmul = numpy_backend.matmul
        original_gradient = numpy_backend.matmul_gradient

        def matmul_spy(left, right, *args, **kwargs):
            self.assertIsInstance(left, numpy.ndarray)
            self.assertIsInstance(right, numpy.ndarray)
            self.assertNotIsInstance(left, ts.Tensor)
            return original_matmul(left, right, *args, **kwargs)

        def gradient_spy(grad, left, right, *args, **kwargs):
            for value in (grad, left, right):
                self.assertIsInstance(value, numpy.ndarray)
                self.assertNotIsInstance(value, ts.Tensor)
            return original_gradient(grad, left, right, *args, **kwargs)

        with (
            patch.object(
                numpy_backend, "matmul", side_effect=matmul_spy
            ) as matmul_kernel,
            patch.object(
                numpy_backend, "matmul_gradient", side_effect=gradient_spy
            ) as gradient_kernel,
            patch.object(
                python_backend,
                "matmul",
                side_effect=AssertionError("Python fallback executed"),
            ),
            patch.object(
                python_backend,
                "matmul_gradient",
                side_effect=AssertionError("Python fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            left = ts.Variable(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))
            right = ts.Variable(ts.Tensor([[5.0, 6.0], [7.0, 8.0]]))
            output = ts.matmul(left, right)
            ts.grad(output, (left, right), grad_outputs=ts.full(output.shape, 1.0))
        matmul_kernel.assert_called_once()
        gradient_kernel.assert_called_once()

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_numpy_outer_kernels_receive_native_values_without_python_fallback(self):
        original_outer = numpy_backend.outer
        original_gradient = numpy_backend.outer_gradient

        def outer_spy(left, right, *args, **kwargs):
            self.assertIsInstance(left, numpy.ndarray)
            self.assertIsInstance(right, numpy.ndarray)
            return original_outer(left, right, *args, **kwargs)

        def gradient_spy(grad, left, right, *args, **kwargs):
            for value in (grad, left, right):
                self.assertIsInstance(value, numpy.ndarray)
            return original_gradient(grad, left, right, *args, **kwargs)

        with (
            patch.object(numpy_backend, "outer", side_effect=outer_spy) as outer_kernel,
            patch.object(
                numpy_backend, "outer_gradient", side_effect=gradient_spy
            ) as gradient_kernel,
            patch.object(
                python_backend,
                "outer",
                side_effect=AssertionError("Python fallback executed"),
            ),
            patch.object(
                python_backend,
                "outer_gradient",
                side_effect=AssertionError("Python fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            left = ts.Variable(ts.Tensor([1.0, 2.0]))
            right = ts.Variable(ts.Tensor([3.0, 4.0]))
            output = ts.outer(left, right)
            ts.grad(output, (left, right), grad_outputs=ts.full(output.shape, 1.0))
        outer_kernel.assert_called_once()
        gradient_kernel.assert_called_once()

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_unrequested_operand_vjps_are_skipped(self):
        with (
            patch.object(
                numpy_backend,
                "matmul_gradient",
                wraps=numpy_backend.matmul_gradient,
            ) as matmul_gradient,
            patch.object(
                numpy_backend,
                "outer_gradient",
                wraps=numpy_backend.outer_gradient,
            ) as outer_gradient,
            ts.use_backend("numpy"),
        ):
            matrix = ts.Variable(ts.Tensor([[1.0, 2.0], [3.0, 4.0]]))
            constant_matrix = ts.Tensor([[5.0, 6.0], [7.0, 8.0]])
            product = ts.matmul(matrix, constant_matrix)
            ts.grad(product, matrix, grad_outputs=ts.full(product.shape, 1.0))

            vector = ts.Variable(ts.Tensor([1.0, 2.0]))
            constant_vector = ts.Tensor([3.0, 4.0])
            product = ts.outer(vector, constant_vector)
            ts.grad(product, vector, grad_outputs=ts.full(product.shape, 1.0))

        self.assertEqual(
            matmul_gradient.call_args.kwargs["needs_input_grad"], (True, False)
        )
        self.assertEqual(
            outer_gradient.call_args.kwargs["needs_input_grad"], (True, False)
        )

    @unittest.skipUnless("numpy" in ts.available_backends(), "NumPy unavailable")
    def test_kernel_decline_is_reported_without_fallback(self):
        with (
            patch.object(numpy_backend, "matmul", return_value=None),
            patch.object(
                python_backend,
                "matmul",
                side_effect=AssertionError("Python fallback executed"),
            ),
            ts.use_backend("numpy"),
        ):
            with self.assertRaises(BackendOperationUnsupportedError):
                ts.matmul(ts.Tensor([[1.0]]), ts.Tensor([[2.0]]))

    def test_accelerated_numerical_kernels_have_no_python_loops_or_host_reads(self):
        repository = Path(__file__).resolve().parents[3]
        for backend in ("numpy", "cuda"):
            for filename in (
                "contraction.py",
                "matmul.py",
                "matmul_gradient.py",
                "outer.py",
                "outer_gradient.py",
            ):
                with self.subTest(backend=backend, filename=filename):
                    source = (
                        repository
                        / "tensors"
                        / "backend"
                        / backend
                        / "kernels"
                        / "linalg"
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
                        "._data",
                        ".tolist(",
                        ".get(",
                        "asnumpy",
                        "tensor_to_logical_array",
                    ):
                        self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
