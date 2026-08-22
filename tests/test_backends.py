import threading
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
import tensors.backend.numpy as numpy_backend


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_python_backend_is_always_available(self):
        self.assertIn("python", ts.available_backends())

    def test_set_backend_changes_process_default(self):
        ts.set_backend("python")

        self.assertEqual(ts.get_backend(), "python")

    def test_auto_backend_falls_back_to_python(self):
        with patch.object(backend_state, "_numpy_available", return_value=False):
            ts.set_backend("auto")

        self.assertEqual(ts.get_backend(), "python")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            ts.set_backend("missing")  # type: ignore[arg-type]

    def test_unavailable_numpy_backend_has_install_guidance(self):
        with patch.object(backend_state, "_numpy_available", return_value=False):
            with self.assertRaisesRegex(
                ts.BackendUnavailableError,
                r"ms-tensors\[numpy\]",
            ):
                ts.set_backend("numpy")

    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_is_nested_and_restored(self):
        ts.set_backend("python")

        with ts.use_backend("numpy"):
            self.assertEqual(ts.get_backend(), "numpy")
            with ts.use_backend("python"):
                self.assertEqual(ts.get_backend(), "python")
            self.assertEqual(ts.get_backend(), "numpy")

        self.assertEqual(ts.get_backend(), "python")

    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_is_restored_after_an_error(self):
        ts.set_backend("python")

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with ts.use_backend("numpy"):
                raise RuntimeError("stop")

        self.assertEqual(ts.get_backend(), "python")

    @unittest.skipUnless(
        "numpy" in ts.available_backends(),
        "NumPy is not installed",
    )
    def test_context_override_does_not_replace_worker_default(self):
        ts.set_backend("python")
        observed = []

        with ts.use_backend("numpy"):
            worker = threading.Thread(
                target=lambda: observed.append(ts.get_backend()),
            )
            worker.start()
            worker.join()
            self.assertEqual(ts.get_backend(), "numpy")

        self.assertEqual(observed, ["python"])


@unittest.skipUnless(
    "numpy" in ts.available_backends(),
    "NumPy is not installed",
)
class NumPyBackendTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _matmul(self, backend, left, right):
        with ts.use_backend(backend):
            return ts.matmul(left, right)

    def assertBackendParity(self, left, right):
        expected = self._matmul("python", left, right)
        actual = self._matmul("numpy", left, right)

        self.assertEqual(actual.shape, expected.shape)
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.tolist(), expected.tolist())
    def _evaluate(self, backend, function):
        with ts.use_backend(backend):
            result = function()
            return result.data if isinstance(result, ts.Variable) else result

    def assertOperationParity(self, function):
        expected = self._evaluate("python", function)
        actual = self._evaluate("numpy", function)

        self.assertEqual(actual.shape, expected.shape)
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.tolist(), expected.tolist())

    def test_every_binary_operation_dispatches_to_numpy(self):
        left = ts.Tensor([[1.0], [2.0]])
        right = ts.Tensor([[3.0, 4.0]])

        with patch.object(
            numpy_backend,
            "binary",
            wraps=numpy_backend.binary,
        ) as binary:
            with ts.use_backend("numpy"):
                _ = left + right
                _ = left - right
                _ = left * right
                _ = left / right
                _ = left ** right
                _ = 2.0 / left
                _ = 2.0 ** left

        self.assertEqual(
            [call.args[0] for call in binary.call_args_list],
            [
                "add",
                "subtract",
                "multiply",
                "divide",
                "power",
                "divide",
                "power",
            ],
        )

    def test_negation_dispatches_to_numpy(self):
        with patch.object(
            numpy_backend,
            "negate",
            wraps=numpy_backend.negate,
        ) as negate:
            self._evaluate("numpy", lambda: -ts.Tensor([1.0, -2.0]))

        negate.assert_called_once()

    def test_slice_dispatches_to_numpy(self):
        value = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
        with patch.object(
            numpy_backend,
            "slice_tensor",
            wraps=numpy_backend.slice_tensor,
        ) as slice_tensor:
            self._evaluate("numpy", lambda: value[:, 1])

        slice_tensor.assert_called_once()

    def test_slice_scatter_dispatches_to_numpy(self):
        with patch.object(
            numpy_backend,
            "slice_scatter",
            wraps=numpy_backend.slice_scatter,
        ) as slice_scatter:
            with ts.use_backend("numpy"):
                value = ts.Variable([1.0, 2.0, 3.0])
                output = ts.sum(value[1:])
                ts.grad(output, value, create_graph=True)

        slice_scatter.assert_called_once()

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
                denominator = ts.Variable([2.0])
                ts.grad(3.0 / denominator, denominator, create_graph=True)

                base = ts.Variable([2.0])
                exponent = ts.Variable([3.0])
                output = base ** exponent
                ts.grad(output, base, create_graph=True)
                ts.grad(output, exponent, create_graph=True)

        division_gradient.assert_called_once()
        self.assertGreaterEqual(power_base_gradient.call_count, 1)
        self.assertGreaterEqual(power_exponent_gradient.call_count, 1)

    def test_cast_dispatches_to_numpy(self):
        value = ts.Tensor([1.25, -2.75])
        with patch.object(
            numpy_backend,
            "cast_tensor",
            wraps=numpy_backend.cast_tensor,
        ) as cast_tensor:
            self._evaluate("numpy", lambda: value.astype(ts.int32))

        cast_tensor.assert_called_once()

    def test_reductions_dispatch_to_numpy(self):
        value = ts.Tensor([float(index + 1) for index in range(512)])
        with patch.object(
            numpy_backend,
            "reduction",
            wraps=numpy_backend.reduction,
        ) as reduction:
            with ts.use_backend("numpy"):
                ts.sum(value)
                ts.mean(value)
                ts.variance(value)
                ts.norm(value)

        self.assertEqual(
            [call.args[0] for call in reduction.call_args_list],
            ["sum", "mean", "variance", "norm"],
        )

    def test_axis_reductions_match_python_backend(self):
        value = ts.Tensor(
            [float(index + 1) for index in range(24)],
            shape=(2, 3, 4),
        )

        for operation in (ts.sum, ts.mean, ts.variance, ts.norm):
            with self.subTest(operation=operation.__name__):
                self.assertOperationParity(
                    lambda operation=operation: operation(
                        value,
                        axis=(0, 2),
                        keepdims=True,
                    )
                )

    def test_stable_reductions_fall_back_without_changing_results(self):
        value = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])
        smallest = ts.Tensor([5e-324, 5e-324])

        self.assertOperationParity(lambda: ts.sum(value))
        self.assertOperationParity(lambda: ts.mean(smallest))
        self.assertOperationParity(lambda: ts.variance(value))
        self.assertOperationParity(lambda: ts.norm(value))

    def test_broadcast_arithmetic_matches_python_backend(self):
        left = ts.Tensor([[1.5], [2.5]])
        right = ts.Tensor([[3.0, 4.0]])

        for operation in (
            lambda: left + right,
            lambda: left - right,
            lambda: left * right,
            lambda: left / right,
            lambda: left ** right,
        ):
            with self.subTest(operation=operation):
                self.assertOperationParity(operation)

    def test_reverse_scalar_operations_match_python_backend(self):
        value = ts.Tensor([1.0, 2.0, 4.0])

        self.assertOperationParity(lambda: 8.0 / value)
        self.assertOperationParity(lambda: 2.0 ** value)

    def test_numpy_division_validates_tensor_denominators_in_kernel(self):
        numerator = ts.Tensor([1.0] * 512)
        denominator = ts.Tensor([2.0] * 511 + [0.0])

        with patch.object(
            numpy_backend,
            "binary",
            wraps=numpy_backend.binary,
        ) as binary:
            with ts.use_backend("numpy"):
                with self.assertRaisesRegex(ZeroDivisionError, "Division by zero"):
                    ts.divide(numerator, denominator)

        binary.assert_called_once()

    def test_integer_arithmetic_and_unsigned_negation_match(self):
        left = ts.Tensor([1, 2, 3], dtype=ts.int32)
        right = ts.Tensor([4, 5, 6], dtype=ts.int32)
        unsigned = ts.Tensor([1, 255], dtype=ts.uint8)

        self.assertOperationParity(lambda: left + right)
        self.assertOperationParity(lambda: left * right)
        self.assertOperationParity(lambda: left ** 3)
        self.assertOperationParity(lambda: -unsigned)

    def test_slice_and_cast_match_python_backend(self):
        value = ts.Variable([[1.25, 2.75], [3.5, 4.125]])

        self.assertOperationParity(lambda: value[:, 1])
        self.assertOperationParity(
            lambda: value.astype(ts.float32),
        )

    def test_numpy_kernel_is_used_for_floating_point_matmul(self):
        import numpy

        original_matmul = numpy.matmul
        with patch.object(numpy, "matmul", wraps=original_matmul) as matmul:
            self._matmul(
                "numpy",
                ts.Tensor([[1.0, 2.0]]),
                ts.Tensor([[3.0], [4.0]]),
            )

        matmul.assert_called_once()

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

    def test_backward_matches_python_backend(self):
        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
                right = ts.Variable([[5.0], [6.0]])
                ts.backward(ts.sum(left @ right))
                return left.grad.tolist(), right.grad.tolist()

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
