import math
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend.numpy as numpy_backend
from tensors.storage import CudaStorage

from ._support import NumPyParityTestCase, requires_cuda, requires_numpy


@requires_cuda
class CudaExtremeGradientTests(unittest.TestCase):
    """Extreme-magnitude VJPs stay device-resident."""

    def test_extreme_vjps_remain_device_resident(self):
        with ts.use_backend("cuda"):
            numerator = ts.Variable(ts.full((64,), 1.0e308))
            denominator = ts.Variable(ts.full((64,), 1.0e308))
            division_gradient = ts.grad(
                numerator / denominator,
                denominator,
                ts.full((64,), 1.0e308),
            )

            base = ts.Variable(ts.full((64,), 1.0e-308))
            base_gradient = ts.grad(
                base ** 2.0,
                base,
                ts.full((64,), 1.0e308),
            )

            exponent_base = ts.Variable(ts.full((64,), 1.0e-200))
            exponent = ts.Variable(ts.full((64,), 3.0))
            exponent_gradient = ts.grad(
                exponent_base ** exponent,
                exponent,
                ts.full((64,), 1.0e308),
            )

            broadcast_value = ts.Variable([0.0])
            factor = ts.Variable(
                [1.0e308, -1.0e308],
                requires_grad=False,
            )
            cancellation = ts.grad(
                broadcast_value * factor,
                broadcast_value,
                ts.Tensor([2.0, 2.0]),
            )

            reduction = ts.sum(ts.Tensor([
                1.0e308,
                1.0e308,
                -1.0e308,
                -1.0e308,
            ]))
            matrix_product = ts.Tensor([
                1.0e308,
                1.0e308,
                -1.0e308,
                -1.0e308,
            ]) @ ts.ones((4,))
            batched_left = ts.Variable(ts.Tensor(
                [1.0e308, 1.0e308, -1.0e308, -1.0e308],
                shape=(4, 1, 1),
            ))
            shared_right = ts.Variable([[1.0]])
            matrix_gradient = ts.grad(
                batched_left @ shared_right,
                shared_right,
                ts.ones((4, 1, 1)),
            )

        for gradient in (
            division_gradient,
            base_gradient,
            exponent_gradient,
            cancellation,
            reduction,
            matrix_product,
            matrix_gradient,
        ):
            self.assertIsInstance(gradient._storage, CudaStorage)
        self.assertEqual(division_gradient[0], -1.0)
        self.assertAlmostEqual(base_gradient[0], 2.0, places=12)
        self.assertTrue(math.isclose(
            exponent_gradient[0],
            -4.605170185988183e-290,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ))
        self.assertEqual(cancellation.tolist(), [0.0])
        self.assertEqual(reduction.tolist(), [0.0])
        self.assertEqual(matrix_product.item(), 0.0)
        self.assertEqual(matrix_gradient.tolist(), [0.0])


@requires_numpy
class NumPyDerivativeHelperTests(NumPyParityTestCase):
    """Division and power VJP helper kernels."""

    def test_derivative_helper_operations_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend,
                "division_denominator_gradient",
                wraps=numpy_backend.division_denominator_gradient,
            ) as division_gradient,
            patch.object(
                numpy_backend,
                "power_base_gradient",
                wraps=numpy_backend.power_base_gradient,
            ) as power_base_gradient,
            patch.object(
                numpy_backend,
                "power_exponent_gradient",
                wraps=numpy_backend.power_exponent_gradient,
            ) as power_exponent_gradient,
        ):
            with ts.use_backend("numpy"):
                denominator = ts.Variable(ts.full((64,), 2.0))
                ts.grad(
                    ts.sum(3.0 / denominator),
                    denominator,
                    create_graph=True,
                )

                base = ts.Variable(ts.full((64,), 2.0))
                exponent = ts.Variable(ts.full((64,), 3.0))
                output = ts.sum(base ** exponent)
                ts.grad(output, base, create_graph=True)
                ts.grad(output, exponent, create_graph=True)

        division_gradient.assert_called_once()
        self.assertGreaterEqual(power_base_gradient.call_count, 1)
        self.assertGreaterEqual(power_exponent_gradient.call_count, 1)


if __name__ == "__main__":
    unittest.main()
