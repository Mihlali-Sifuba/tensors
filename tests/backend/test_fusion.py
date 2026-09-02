import math
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
import tensors.backend.cuda as cuda_backend
from tensors.storage import CudaStorage, PythonStorage

from ._support import requires_cuda


@requires_cuda
class CudaFusionTests(unittest.TestCase):
    """Elementwise chains fuse into one CUDA launch in both directions."""

    def test_graph_replay_fuses_scalar_elementwise_chains(self):
        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 1.0), requires_grad=False)
            intermediate = value * 2.0
            output = intermediate + 3.0
            computation = ts.graph.Computation(output)
            value.data = ts.full((4_096,), 4.0)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fused:
                result = computation.forward()

        fused.assert_called_once()
        self.assertIsInstance(result._storage, CudaStorage)
        self.assertEqual(intermediate.data.tolist(), [8.0] * 4_096)
        self.assertEqual(result.tolist(), [11.0] * 4_096)
    def test_backward_fuses_scalar_elementwise_chains(self):
        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 2.0))
            intermediate = value * 2.0
            output = intermediate * 3.0 + 1.0
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as fused:
                backend_state._clear_backend_kernel_cache()
                computation.backward(ts.full((4_096,), 1.0))

        fused.assert_called_once()
        self.assertIsInstance(value.grad._storage, CudaStorage)
        self.assertEqual(intermediate.grad.tolist(), [3.0] * 4_096)
        self.assertEqual(value.grad.tolist(), [6.0] * 4_096)
    def test_graph_fusion_supports_every_floating_dtype(self):
        for dtype in (ts.float32, ts.float64):
            with self.subTest(dtype=dtype.name), ts.use_backend("cuda"):
                value = ts.Variable(ts.full((4_096,), 0.25, dtype=dtype))
                exponential = ts.exp(value)
                probability = ts.sigmoid(exponential)
                activation = ts.tanh(probability)
                output = ts.softplus(activation)
                computation = ts.graph.Computation(output)
                with patch.object(
                    cuda_backend,
                    "fused_elementwise",
                    wraps=cuda_backend.fused_elementwise,
                ) as forward_fusion:
                    backend_state._clear_backend_kernel_cache()
                    result = computation.forward()
                with patch.object(
                    cuda_backend,
                    "fused_elementwise_backward",
                    wraps=cuda_backend.fused_elementwise_backward,
                ) as backward_fusion:
                    backend_state._clear_backend_kernel_cache()
                    computation.backward(ts.full((4_096,), 1.0, dtype=dtype))

                expected_exp = math.exp(0.25)
                expected_sigmoid = 1.0 / (1.0 + math.exp(-expected_exp))
                expected_tanh = math.tanh(expected_sigmoid)
                expected = math.log1p(math.exp(expected_tanh))
                expected_gradient = (
                    expected_exp
                    * expected_sigmoid
                    * (1.0 - expected_sigmoid)
                    * (1.0 - expected_tanh * expected_tanh)
                    / (1.0 + math.exp(-expected_tanh))
                )

                forward_fusion.assert_called_once()
                backward_fusion.assert_called_once()
                self.assertEqual(result.dtype, dtype)
                self.assertIsInstance(result._storage, CudaStorage)
                self.assertIsInstance(value.grad._storage, CudaStorage)
                self.assertAlmostEqual(
                    float(result[0]),
                    expected,
                    places=5 if dtype == ts.float32 else 12,
                )
                self.assertAlmostEqual(
                    float(value.grad[0]),
                    expected_gradient,
                    places=5 if dtype == ts.float32 else 12,
                )
    def test_graph_fusion_supports_tensor_arithmetic_and_vjps(self):
        with ts.use_backend("cuda"):
            left = ts.Variable(ts.full((4_096,), 2.0))
            right = ts.Variable(ts.full((4_096,), 3.0))
            bias = ts.Variable(ts.full((4_096,), 10.0))
            product = left * right
            shifted = bias - product
            output = ts.relu(shifted)
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as fused:
                backend_state._clear_backend_kernel_cache()
                computation.backward(ts.full((4_096,), 1.0))

        fused.assert_called_once()
        self.assertEqual(product.grad.tolist(), [-1.0] * 4_096)
        self.assertEqual(shifted.grad.tolist(), [1.0] * 4_096)
        self.assertEqual(left.grad.tolist(), [-3.0] * 4_096)
        self.assertEqual(right.grad.tolist(), [-2.0] * 4_096)
        self.assertEqual(bias.grad.tolist(), [1.0] * 4_096)
    def test_graph_replay_fuses_post_matmul_bias_and_activation(self):
        with ts.use_backend("cuda"):
            value = ts.Variable(
                ts.full((64, 64), 1.0),
                requires_grad=False,
            )
            weight = ts.Variable(
                ts.full((64, 64), 1.0),
                requires_grad=False,
            )
            bias = ts.Variable(ts.full((64,), 0.5), requires_grad=False)
            product = value @ weight
            shifted = product + bias
            output = ts.relu(shifted)
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fused:
                backend_state._clear_backend_kernel_cache()
                result = computation.forward()

        fused.assert_called_once()
        self.assertIsInstance(result._storage, CudaStorage)
        self.assertAlmostEqual(float(result[0, 0]), 64.5)
        self.assertAlmostEqual(float(shifted.data[0, 0]), 64.5)
    def test_fused_tensor_division_preserves_zero_validation(self):
        with ts.use_backend("cuda"):
            numerator = ts.Variable(
                ts.full((4_096,), 2.0),
                requires_grad=False,
            )
            denominator = ts.Variable(
                ts.full((4_096,), 1.0),
                requires_grad=False,
            )
            output = ts.relu(numerator / denominator)
            computation = ts.graph.Computation(output)
            denominator.data = ts.zeros((4_096,))

            with self.assertRaisesRegex(ZeroDivisionError, "Division by zero"):
                computation.forward()
    def test_extended_unary_and_power_chains_fuse_in_both_directions(self):
        def expression(value):
            return ts.sin(ts.log(ts.sqrt(value + 2.0))) ** 2.0

        with ts.use_backend("python"):
            reference_input = ts.Variable(ts.full((4_096,), 0.25))
            reference_output = expression(reference_input)
            reference_gradient = ts.grad(
                reference_output,
                reference_input,
                ts.ones((4_096,)),
            )

        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 0.25))
            output = expression(value)
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as forward_fusion:
                backend_state._clear_backend_kernel_cache()
                result = computation.forward()
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as backward_fusion:
                backend_state._clear_backend_kernel_cache()
                computation.backward(ts.ones((4_096,)))

        forward_fusion.assert_called_once()
        backward_fusion.assert_called_once()
        self.assertIsInstance(result._storage, CudaStorage)
        self.assertIsInstance(value.grad._storage, CudaStorage)
        self.assertAlmostEqual(result[0], reference_output.data[0], places=12)
        self.assertAlmostEqual(value.grad[0], reference_gradient[0], places=12)
    def test_every_extended_unary_operation_fuses(self):
        cases = (
            (ts.sqrt, 2.0),
            (ts.log, 2.0),
            (ts.sin, 0.25),
            (ts.cos, 0.25),
            (ts.tan, 0.25),
            (ts.arcsin, 0.25),
            (ts.arccos, 0.25),
            (ts.arctan, 0.25),
            (ts.sinh, 0.25),
            (ts.cosh, 0.25),
            (ts.arcsinh, 0.25),
            (ts.arccosh, 2.0),
            (ts.arctanh, 0.25),
            (ts.sign, 0.25),
        )
        for operation, input_value in cases:
            with self.subTest(operation=operation.__name__):
                with ts.use_backend("python"):
                    reference = ts.Variable([input_value])
                    reference_output = operation(reference) + 1.0
                    reference_gradient = ts.grad(reference_output, reference)

                with ts.use_backend("cuda"):
                    value = ts.Variable(ts.full((4_096,), input_value))
                    output = operation(value) + 1.0
                    computation = ts.graph.Computation(output)
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise",
                        wraps=cuda_backend.fused_elementwise,
                    ) as forward_fusion:
                        backend_state._clear_backend_kernel_cache()
                        result = computation.forward()
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise_backward",
                        wraps=cuda_backend.fused_elementwise_backward,
                    ) as backward_fusion:
                        backend_state._clear_backend_kernel_cache()
                        computation.backward(ts.ones((4_096,)))

                forward_fusion.assert_called_once()
                backward_fusion.assert_called_once()
                self.assertAlmostEqual(result[0], reference_output.data[0], places=12)
                self.assertAlmostEqual(value.grad[0], reference_gradient[0], places=12)
    def test_fused_backward_reduces_broadcast_tensor_division_vjps(self):
        with ts.use_backend("cuda"):
            numerator = ts.Variable(ts.full((4_096, 1), 2.0))
            denominator = ts.Variable(ts.Tensor([[1.0, 2.0, 4.0, 8.0]]))
            quotient = numerator / denominator
            output = quotient + 1.0
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as fused:
                backend_state._clear_backend_kernel_cache()
                computation.backward(ts.ones(output.shape))

        fused.assert_called_once()
        self.assertIsInstance(numerator.grad._storage, CudaStorage)
        self.assertIsInstance(denominator.grad._storage, CudaStorage)
        expected_numerator = 1.0 + 0.5 + 0.25 + 0.125
        self.assertAlmostEqual(numerator.grad[0, 0], expected_numerator)
        expected_denominator = [-8_192.0, -2_048.0, -512.0, -128.0]
        self.assertEqual(denominator.grad.tolist(), expected_denominator)
    def test_fused_backward_supports_reverse_division(self):
        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 4.0))
            reciprocal = 2.0 / value
            output = reciprocal + 1.0
            computation = ts.graph.Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as fused:
                backend_state._clear_backend_kernel_cache()
                computation.backward(ts.ones((4_096,)))

        fused.assert_called_once()
        self.assertIsInstance(value.grad._storage, CudaStorage)
        self.assertEqual(value.grad.tolist(), [-0.125] * 4_096)
    def test_fused_extreme_power_and_division_vjps_retain_range(self):
        with ts.use_backend("cuda"):
            base = ts.Variable(ts.full((4_096,), 1.0e-200))
            powered = base ** 3.0
            power_output = powered + 1.0
            power_computation = ts.graph.Computation(power_output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as power_fusion:
                backend_state._clear_backend_kernel_cache()
                power_computation.backward(ts.full((4_096,), 1.0e308))

            numerator = ts.Variable(ts.full((4_096,), 1.0e308))
            denominator = ts.Variable(ts.full((4_096,), 1.0e308))
            quotient = numerator / denominator
            division_output = quotient + 1.0
            division_computation = ts.graph.Computation(division_output)
            with patch.object(
                cuda_backend,
                "fused_elementwise_backward",
                wraps=cuda_backend.fused_elementwise_backward,
            ) as division_fusion:
                backend_state._clear_backend_kernel_cache()
                division_computation.backward(ts.full((4_096,), 1.0e308))

        power_fusion.assert_called_once()
        division_fusion.assert_called_once()
        self.assertIsInstance(base.grad._storage, CudaStorage)
        self.assertIsInstance(denominator.grad._storage, CudaStorage)
        self.assertTrue(math.isclose(
            base.grad[0],
            3.0e-92,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ))
        self.assertEqual(denominator.grad[0], -1.0)
    def test_fused_replay_preserves_extended_math_domains(self):
        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 1.0), requires_grad=False)
            root = ts.sqrt(value)
            output = root + 1.0
            computation = ts.graph.Computation(output)
            value.data = ts.full((4_096,), -1.0)

            with self.assertRaisesRegex(ValueError, "sqrt"):
                computation.forward()
    def test_integer_graphs_keep_the_exact_reference_path(self):
        integer_dtypes = (
            ts.int64,
            ts.int32,
            ts.int16,
            ts.int8,
            ts.uint8,
        )
        for dtype in integer_dtypes:
            with self.subTest(dtype=dtype.name), ts.use_backend("cuda"):
                value = ts.Variable(
                    ts.full((4_096,), 2, dtype=dtype),
                    requires_grad=False,
                )
                output = (value + 1) * 2
                computation = ts.graph.Computation(output)
                with patch.object(
                    cuda_backend,
                    "fused_elementwise",
                    wraps=cuda_backend.fused_elementwise,
                ) as fused:
                    backend_state._clear_backend_kernel_cache()
                    result = computation.forward()

                fused.assert_not_called()
                self.assertIsInstance(result._storage, PythonStorage)
                self.assertEqual(result[0], 6)


if __name__ == "__main__":
    unittest.main()
