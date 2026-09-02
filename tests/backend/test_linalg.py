import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend
from tensors.storage import CudaStorage

from ._support import NumPyParityTestCase, requires_cuda, requires_numpy


@requires_cuda
class CudaMatmulTests(unittest.TestCase):
    """Device matrix products agree with the Python reference."""

    def test_cuda_matmul_matches_python(self):
        left = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        right = ts.Tensor([[2.0, 0.0], [1.0, 2.0]])
        with ts.use_backend("python"):
            expected = (left @ right).tolist()
        with ts.use_backend("cuda"):
            actual = left @ right

        self.assertIsInstance(actual._storage, CudaStorage)
        self.assertEqual(actual.tolist(), expected)


@requires_numpy
class NumPyMatmulTests(NumPyParityTestCase):
    """Matrix-product kernels, their VJPs, and their fallbacks."""

    def test_numpy_kernel_is_used_for_floating_point_matmul(self):
        import numpy

        original_matmul = numpy.matmul
        with patch.object(numpy, "matmul", wraps=original_matmul) as matmul:
            self._matmul(
                "numpy",
                ts.full((4, 4), 2.0),
                ts.full((4, 4), 3.0),
            )

        matmul.assert_called_once()
    def test_numpy_kernel_is_used_for_floating_point_matmul_gradient(self):
        with patch.object(
            numpy_backend,
            "matmul_gradient",
            wraps=numpy_backend.matmul_gradient,
        ) as matmul_gradient:
            with ts.use_backend("numpy"):
                left = ts.Variable(ts.full((8, 8), 0.25))
                right = ts.Variable(ts.full((8, 8), 0.5))
                ts.backward(ts.sum(left @ right))

        matmul_gradient.assert_called_once()
    def test_broadcast_matmul_gradient_matches_python_backend(self):
        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.full((1, 4, 8), 0.25))
                right = ts.Variable(ts.full((3, 8, 4), 0.5))
                ts.backward(ts.sum(left @ right))
                return left.grad, right.grad

        expected = gradients("python")
        actual = gradients("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            for actual_item, expected_item in zip(
                actual_tensor.tolist(),
                expected_tensor.tolist(),
            ):
                self.assertAlmostEqual(actual_item, expected_item)
    def test_vector_product_matches_python_backend(self):
        self.assertBackendParity(
            ts.Tensor([1.0, 2.0, 3.0]),
            ts.Tensor([4.0, 5.0, 6.0]),
        )
    def test_matrix_vector_products_match_python_backend(self):
        matrix = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        vector = ts.Tensor([5.0, 6.0])

        self.assertBackendParity(matrix, vector)
        self.assertBackendParity(vector, matrix)
    def test_batched_broadcast_product_matches_python_backend(self):
        left = ts.Tensor(
            [1.0, 2.0, 3.0, 4.0],
            shape=(1, 2, 2),
        )
        right = ts.Tensor(
            [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
            shape=(2, 2, 2),
        )

        self.assertBackendParity(left, right)
    def test_promoted_float_dtype_matches_python_backend(self):
        self.assertBackendParity(
            ts.Tensor([[1.0, 2.0]], dtype=ts.float32),
            ts.Tensor([[3.0], [4.0]], dtype=ts.float64),
        )
    def test_non_integer_values_agree_within_float_tolerance(self):
        left = ts.Tensor([[0.1, -2.75, 3.125], [4.2, 0.3, -0.625]])
        right = ts.Tensor([[1.2, 0.5], [-0.2, 2.1], [3.4, -1.25]])

        expected = self._matmul("python", left, right)
        actual = self._matmul("numpy", left, right)

        self.assertEqual(actual.shape, expected.shape)
        for actual_value, expected_value in zip(
            actual.tolist(),
            expected.tolist(),
        ):
            self.assertAlmostEqual(actual_value, expected_value, places=12)
    def test_integer_product_uses_compatible_fallback(self):
        self.assertBackendParity(
            ts.Tensor([[1, 2], [3, 4]], dtype=ts.int32),
            ts.Tensor([[5, 6], [7, 8]], dtype=ts.int32),
        )
    def test_temporary_overflow_uses_stable_fallback(self):
        left = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])
        right = ts.Tensor([1.0, 1.0, 1.0, 1.0])

        result = self._matmul("numpy", left, right)

        self.assertEqual(result.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
