"""Numerical certification boundaries for backend-native contractions."""

import math
import unittest

import tensors as ts
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state


class ContractionCertificationTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def assert_accelerated_result_or_unsupported(self, operation, expected):
        for backend in ("numpy", "cuda"):
            if backend not in ts.available_backends():
                continue
            with self.subTest(backend=backend):
                reset_graph_state()
                try:
                    with ts.use_backend(backend):
                        result = operation()
                except BackendOperationUnsupportedError:
                    continue
                self.assertEqual(result, expected)

    def test_hostile_finite_cancellation_is_never_silently_rounded(self):
        left = [1e16, 1.0, -1e16, 1.0]
        right = [1.0, 1.0, 1.0, 1.0]
        with ts.use_backend("python"):
            self.assertEqual(ts.matmul(ts.Tensor(left), ts.Tensor(right)).item(), 2.0)
        self.assert_accelerated_result_or_unsupported(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)).item(),
            2.0,
        )

    def test_individual_product_range_loss_is_explicitly_declined_or_exact(self):
        left = [1e308, -1e308]
        right = [1e308, 1e308]
        with ts.use_backend("python"):
            self.assertEqual(ts.matmul(ts.Tensor(left), ts.Tensor(right)).item(), 0.0)
        self.assert_accelerated_result_or_unsupported(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)).item(),
            0.0,
        )

    def test_matrix_certification_covers_each_output_contraction(self):
        left = [[1.0, 2.0, 3.0, 4.0], [1e16, 1.0, -1e16, 1.0]]
        right = [[1.0], [1.0], [1.0], [1.0]]
        with ts.use_backend("python"):
            expected = ts.matmul(ts.Tensor(left), ts.Tensor(right)).tolist()
        self.assertEqual(expected, [10.0, 2.0])
        self.assert_accelerated_result_or_unsupported(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)).tolist(),
            expected,
        )

    def test_nonfinite_batch_does_not_mask_unsafe_finite_batch(self):
        left = [
            math.inf,
            0.0,
            0.0,
            0.0,
            1e16,
            1.0,
            -1e16,
            1.0,
        ]
        right = [1.0] * 8
        shape = (2, 1, 4)
        right_shape = (2, 4, 1)
        with ts.use_backend("python"):
            expected = ts.matmul(
                ts.Tensor(left, shape=shape),
                ts.Tensor(right, shape=right_shape),
            ).tolist()
        self.assertEqual(expected[1], 2.0)

        def operation():
            result = ts.matmul(
                ts.Tensor(left, shape=shape),
                ts.Tensor(right, shape=right_shape),
            ).tolist()
            self.assertTrue(math.isinf(result[0]))
            return result[1]

        self.assert_accelerated_result_or_unsupported(operation, 2.0)

    def test_hostile_matmul_left_vjp_is_exact_or_unsupported(self):
        upstream = [[1e16, 1.0, -1e16, 1.0]]

        def operation():
            left = ts.Variable(ts.Tensor([[1.0]]))
            right = ts.Tensor([[1.0, 1.0, 1.0, 1.0]])
            output = ts.matmul(left, right)
            return ts.grad(
                output,
                left,
                grad_outputs=ts.Tensor(upstream),
            ).item()

        with ts.use_backend("python"):
            self.assertEqual(operation(), 2.0)
        self.assert_accelerated_result_or_unsupported(operation, 2.0)

    def test_hostile_matmul_right_vjp_is_exact_or_unsupported(self):
        upstream = [[1e16], [1.0], [-1e16], [1.0]]

        def operation():
            left = ts.Tensor([[1.0], [1.0], [1.0], [1.0]])
            right = ts.Variable(ts.Tensor([[1.0]]))
            output = ts.matmul(left, right)
            return ts.grad(
                output,
                right,
                grad_outputs=ts.Tensor(upstream),
            ).item()

        with ts.use_backend("python"):
            self.assertEqual(operation(), 2.0)
        self.assert_accelerated_result_or_unsupported(operation, 2.0)

    def test_hostile_outer_left_vjp_is_exact_or_unsupported(self):
        upstream = [[1e16, 1.0, -1e16, 1.0]]

        def operation():
            left = ts.Variable(ts.Tensor([1.0]))
            right = ts.Tensor([1.0, 1.0, 1.0, 1.0])
            output = ts.outer(left, right)
            return ts.grad(
                output,
                left,
                grad_outputs=ts.Tensor(upstream),
            ).item()

        with ts.use_backend("python"):
            self.assertEqual(operation(), 2.0)
        self.assert_accelerated_result_or_unsupported(operation, 2.0)

    def test_hostile_outer_right_vjp_is_exact_or_unsupported(self):
        upstream = [[1e16], [1.0], [-1e16], [1.0]]

        def operation():
            left = ts.Tensor([1.0, 1.0, 1.0, 1.0])
            right = ts.Variable(ts.Tensor([1.0]))
            output = ts.outer(left, right)
            return ts.grad(
                output,
                right,
                grad_outputs=ts.Tensor(upstream),
            ).item()

        with ts.use_backend("python"):
            self.assertEqual(operation(), 2.0)
        self.assert_accelerated_result_or_unsupported(operation, 2.0)


if __name__ == "__main__":
    unittest.main()
