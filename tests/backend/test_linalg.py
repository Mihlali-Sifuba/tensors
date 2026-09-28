import unittest
from unittest.mock import patch
import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.cuda.storage import CudaStorage
from tests.backend._support import NumPyParityTestCase, requires_cuda, requires_numpy


@requires_cuda
class CudaMatmulTests(unittest.TestCase):
    """Device matrix products agree with the Python reference."""

    def test_cuda_matmul_matches_python(self):
        with ts.use_backend("python"):
            expected = (
                ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
                @ ts.Tensor([[2.0, 0.0], [1.0, 2.0]])
            ).tolist()
        with ts.use_backend("cuda"):
            actual = ts.Tensor([[1.0, 2.0], [3.0, 4.0]]) @ ts.Tensor(
                [[2.0, 0.0], [1.0, 2.0]]
            )
        self.assertIsInstance(actual.backend_storage, CudaStorage)
        self.assertEqual(actual.tolist(), expected)


@requires_numpy
class NumPyMatmulTests(NumPyParityTestCase):
    """Matrix-product kernels, their VJPs, and strict declines."""

    def test_numpy_native_certificate_is_used_for_floating_point_matmul(self):
        import tensors.backend.numpy.kernels.linalg.contraction as contraction

        original_sum = contraction.certified_float_sum
        with patch.object(
            contraction, "certified_float_sum", wraps=original_sum
        ) as certified_sum:
            self._matmul("numpy", ts.full((4, 4), 2.0), ts.full((4, 4), 3.0))
        certified_sum.assert_called_once()

    def test_numpy_kernel_is_used_for_floating_point_matmul_gradient(self):
        with patch.object(
            numpy_backend, "matmul_gradient", wraps=numpy_backend.matmul_gradient
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
                return (left.grad, right.grad)

        expected = gradients("python")
        actual = gradients("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            for actual_item, expected_item in zip(
                actual_tensor.tolist(), expected_tensor.tolist()
            ):
                self.assertAlmostEqual(actual_item, expected_item)

    def test_vector_product_matches_python_backend(self):
        self.assertBackendParity(ts.Tensor([1.0, 2.0, 3.0]), ts.Tensor([4.0, 5.0, 6.0]))

    def test_matrix_vector_products_match_python_backend(self):
        matrix = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        vector = ts.Tensor([5.0, 6.0])
        self.assertBackendParity(matrix, vector)
        self.assertBackendParity(vector, matrix)

    def test_batched_broadcast_product_matches_python_backend(self):
        left = ts.Tensor([1.0, 2.0, 3.0, 4.0], shape=(1, 2, 2))
        right = ts.Tensor([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], shape=(2, 2, 2))
        self.assertBackendParity(left, right)

    def test_promoted_float_dtype_matches_python_backend(self):
        self.assertBackendParity(
            ts.Tensor([[1.0, 2.0]], dtype=ts.float32),
            ts.Tensor([[3.0], [4.0]], dtype=ts.float64),
        )

    def test_non_integer_values_are_conforming_or_explicitly_unsupported(self):
        left = ts.Tensor([[0.1, -2.75, 3.125], [4.2, 0.3, -0.625]])
        right = ts.Tensor([[1.2, 0.5], [-0.2, 2.1], [3.4, -1.25]])
        expected = self._matmul("python", left, right)
        try:
            actual = self._matmul("numpy", left, right)
        except BackendOperationUnsupportedError:
            return
        self.assertEqual(actual.shape, expected.shape)
        for actual_value, expected_value in zip(actual.tolist(), expected.tolist()):
            self.assertEqual(actual_value, expected_value)

    def test_integer_product_is_explicitly_unsupported(self):
        with ts.use_backend("numpy"):
            left = ts.Tensor([[1, 2], [3, 4]], dtype=ts.int32)
            right = ts.Tensor([[5, 6], [7, 8]], dtype=ts.int32)
            with self.assertRaises(BackendOperationUnsupportedError):
                ts.matmul(left, right)

    def test_temporary_overflow_is_explicitly_unsupported(self):
        with ts.use_backend("numpy"):
            left = ts.Tensor([1e308, 1e308, -1e308, -1e308])
            right = ts.Tensor([1.0, 1.0, 1.0, 1.0])
            with self.assertRaises(BackendOperationUnsupportedError):
                ts.matmul(left, right)


if __name__ == "__main__":
    unittest.main()
