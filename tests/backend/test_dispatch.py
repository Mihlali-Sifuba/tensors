import unittest
from unittest.mock import patch
import tensors as ts
import tensors.backend.numpy.kernels as numpy_backend
from tests.backend._support import NumPyParityTestCase, requires_numpy


@requires_numpy
class NumPyDispatchBoundaryTests(NumPyParityTestCase):
    """Work-size guards and backend-agnostic graph replay."""

    def test_tiny_non_arithmetic_operations_bypass_numpy_kernel_dispatch(self):
        """The work-size guard still governs operations outside the contract.

        Arithmetic left this guard with breaking change B15: explicit
        selection now decides where ``+`` runs. Negation, casting, reduction
        and matmul are outside the arithmetic contract and keep the guard.
        """
        value = ts.Tensor([2.0])
        with (
            patch.object(numpy_backend, "negate") as negate,
            patch.object(numpy_backend, "cast_tensor") as cast_tensor,
            patch.object(numpy_backend, "reduce_sum") as reduction,
            patch.object(numpy_backend, "matmul") as matmul,
        ):
            with ts.use_backend("numpy"):
                _ = -value
                _ = value.astype(ts.float32)
                _ = ts.sum(value)
                _ = value @ value
        negate.assert_not_called()
        cast_tensor.assert_not_called()
        reduction.assert_not_called()
        matmul.assert_not_called()

    def test_tiny_arithmetic_still_reaches_the_numpy_kernel(self):
        from tensors.backend.loading import load_backend

        backend = load_backend("numpy")
        value = ts.Tensor([2.0])
        with patch.object(backend, "add", wraps=backend.add) as binary:
            with ts.use_backend("numpy"):
                result = value + value
        binary.assert_called_once()
        self.assertEqual(result.tolist(), [4.0])

    def test_backward_matches_python_backend(self):

        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
                right = ts.Variable([[5.0], [6.0]])
                ts.backward(ts.sum(left @ right))
                return (left.grad.tolist(), right.grad.tolist())

        self.assertEqual(gradients("numpy"), gradients("python"))

    def test_recorded_graph_can_replay_with_either_backend(self):
        weights = ts.Tensor([[2.0], [3.0]])

        @ts.Graph
        def model(value):
            return value @ weights

        model(ts.Tensor([[1.0, 2.0]]))
        with ts.use_backend("python"):
            expected = model.computation.forward()
        with ts.use_backend("numpy"):
            actual = model.computation.forward()
        self.assertEqual(actual.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
