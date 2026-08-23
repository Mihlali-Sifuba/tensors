import threading
import unittest
import math
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
import tensors.backend.cuda as cuda_backend
import tensors.backend.numpy as numpy_backend
from tensors.storage import CudaStorage, NumPyStorage, PythonStorage


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

    def test_unavailable_cuda_backend_has_install_guidance(self):
        with patch.object(backend_state, "_cuda_available", return_value=False):
            with self.assertRaisesRegex(
                ts.BackendUnavailableError,
                r"ms-tensors\[cuda1[23]\]",
            ):
                ts.set_backend("cuda")

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
class NativeStorageTests(unittest.TestCase):
    def test_numpy_results_retain_native_storage(self):
        with ts.use_backend("numpy"):
            result = ts.full((64,), 2.0) + 3.0

        self.assertIsInstance(result._storage, NumPyStorage)
        self.assertEqual(result.tolist(), [5.0] * 64)

    def test_backend_views_are_cached_until_mutation(self):
        import numpy

        value = ts.Tensor([1.0, 2.0, 3.0])
        original = value._storage

        first = value._storage_for("numpy")
        second = value._storage_for("numpy")

        self.assertIs(first, second)
        self.assertIs(value._storage, original)
        host_view = numpy.frombuffer(
            original.buffer,
            dtype=numpy.dtype(value.dtype.name),
        )
        self.assertTrue(numpy.shares_memory(first.buffer, host_view))
        value[0] = 4.0
        self.assertIsInstance(value._storage, PythonStorage)
        self.assertEqual(set(value._storage_cache), {"python"})


@unittest.skipUnless(
    "cuda" in ts.available_backends(),
    "CUDA is not available",
)
class CudaBackendTests(unittest.TestCase):
    def test_floating_results_remain_device_resident(self):
        with ts.use_backend("cuda"):
            value = ts.full((64,), 2.0)
            with patch.object(
                cuda_backend,
                "binary",
                wraps=cuda_backend.binary,
            ) as kernel:
                result = value * 3.0 + 1.0

        self.assertGreaterEqual(kernel.call_count, 1)
        self.assertIsInstance(value._storage, CudaStorage)
        self.assertIsInstance(result._storage, CudaStorage)
        self.assertEqual(result.tolist(), [7.0] * 64)

    def test_cuda_matmul_matches_python(self):
        left = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        right = ts.Tensor([[2.0, 0.0], [1.0, 2.0]])
        with ts.use_backend("python"):
            expected = (left @ right).tolist()
        with ts.use_backend("cuda"):
            actual = left @ right

        self.assertIsInstance(actual._storage, CudaStorage)
        self.assertEqual(actual.tolist(), expected)

    def test_integer_operations_use_reference_storage(self):
        with ts.use_backend("cuda"):
            result = ts.full((64,), 2, dtype=ts.int32) + 3

        self.assertIsInstance(result._storage, PythonStorage)
        self.assertEqual(result.tolist(), [5] * 64)

    def test_optimizer_updates_remain_device_resident(self):
        with ts.use_backend("cuda"):
            parameter = ts.Variable(ts.full((64,), 1.0))
            parameter.grad = ts.full((64,), 0.5)
            ts.optim.SGD([parameter], learning_rate=0.1).step()

        self.assertIsInstance(parameter.data._storage, CudaStorage)
        self.assertAlmostEqual(parameter.data[0], 0.95)

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

    def test_multi_parameter_sgd_uses_grouped_cuda_update(self):
        with ts.use_backend("cuda"):
            parameters = [
                ts.Variable(ts.full((64,), float(index + 1)))
                for index in range(4)
            ]
            for parameter in parameters:
                parameter.grad = ts.full((64,), 0.5)
            optimizer = ts.optim.SGD(parameters, learning_rate=0.1)
            with patch.object(
                cuda_backend,
                "sgd_updates",
                wraps=cuda_backend.sgd_updates,
            ) as grouped:
                optimizer.step()

        grouped.assert_called_once()
        self.assertTrue(all(
            isinstance(parameter.data._storage, CudaStorage)
            for parameter in parameters
        ))


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

    def test_multi_parameter_optimizers_use_grouped_updates(self):
        optimizers = (
            (ts.optim.SGD, "sgd_updates"),
            (ts.optim.Adam, "adam_updates"),
            (ts.optim.RMSprop, "rmsprop_updates"),
        )
        for optimizer_type, kernel_name in optimizers:
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("numpy"):
                    parameters = [
                        ts.Variable(ts.full((64,), float(index + 1)))
                        for index in range(4)
                    ]
                    for parameter in parameters:
                        parameter.grad = ts.full((64,), 0.5)
                    optimizer = optimizer_type(
                        parameters,
                        learning_rate=0.01,
                    )
                    kernel = getattr(numpy_backend, kernel_name)
                    with patch.object(
                        numpy_backend,
                        kernel_name,
                        wraps=kernel,
                    ) as grouped:
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()

                grouped.assert_called_once()
                self.assertTrue(all(
                    isinstance(parameter.data._storage, NumPyStorage)
                    for parameter in parameters
                ))

    def test_grouped_optimizer_updates_match_python(self):
        def updated(backend, optimizer_type):
            with ts.use_backend(backend):
                parameters = [
                    ts.Variable(ts.full((64,), float(index + 1)))
                    for index in range(4)
                ]
                for index, parameter in enumerate(parameters):
                    parameter.grad = ts.full(
                        (64,),
                        0.25 * (index + 1),
                    )
                optimizer = optimizer_type(
                    parameters,
                    learning_rate=0.01,
                )
                optimizer.step()
                optimizer.step()
                return tuple(parameter.data.tolist() for parameter in parameters)

        for optimizer_type in (ts.optim.SGD, ts.optim.Adam, ts.optim.RMSprop):
            with self.subTest(optimizer=optimizer_type.__name__):
                expected = updated("python", optimizer_type)
                actual = updated("numpy", optimizer_type)
                for expected_values, actual_values in zip(expected, actual):
                    for expected_value, actual_value in zip(
                        expected_values,
                        actual_values,
                    ):
                        self.assertAlmostEqual(actual_value, expected_value)

    def test_every_binary_operation_dispatches_to_numpy(self):
        left = ts.full((32, 1), 2.0)
        right = ts.full((1, 32), 3.0)

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
            self._evaluate("numpy", lambda: -ts.full((64,), 2.0))

        negate.assert_called_once()

    def test_every_unary_operation_dispatches_to_numpy(self):
        operations = (
            ("abs", ts.abs, 0.5),
            ("sqrt", ts.sqrt, 0.5),
            ("exp", ts.exp, 0.5),
            ("log", ts.log, 0.5),
            ("sin", ts.sin, 0.5),
            ("cos", ts.cos, 0.5),
            ("tan", ts.tan, 0.5),
            ("arcsin", ts.arcsin, 0.5),
            ("arccos", ts.arccos, 0.5),
            ("arctan", ts.arctan, 0.5),
            ("sinh", ts.sinh, 0.5),
            ("cosh", ts.cosh, 0.5),
            ("arcsinh", ts.arcsinh, 0.5),
            ("arccosh", ts.arccosh, 1.5),
            ("arctanh", ts.arctanh, 0.5),
            ("sign", ts.sign, 0.5),
            ("relu", ts.relu, 0.5),
            ("sigmoid", ts.sigmoid, 0.5),
            ("tanh", ts.tanh, 0.5),
            ("softplus", ts.softplus, 0.5),
        )
        with (
            patch.object(
                numpy_backend,
                "unary",
                wraps=numpy_backend.unary,
            ) as unary,
            patch.object(
                numpy_backend,
                "unary_gradient",
                wraps=numpy_backend.unary_gradient,
            ) as unary_gradient,
        ):
            with ts.use_backend("numpy"):
                for _, function, item in operations:
                    value = ts.Variable(ts.full((64,), item))
                    output = function(value)
                    ts.grad(ts.sum(output), value)

        expected = [name for name, _, _ in operations]
        self.assertEqual(
            [call.args[0] for call in unary.call_args_list],
            expected,
        )
        self.assertEqual(
            [call.args[0] for call in unary_gradient.call_args_list],
            expected,
        )

    def test_unary_operations_and_gradients_match_python_backend(self):
        operations = (
            (ts.abs, 0.5),
            (ts.sqrt, 0.5),
            (ts.exp, 0.5),
            (ts.log, 0.5),
            (ts.sin, 0.5),
            (ts.cos, 0.5),
            (ts.tan, 0.5),
            (ts.arcsin, 0.5),
            (ts.arccos, 0.5),
            (ts.arctan, 0.5),
            (ts.sinh, 0.5),
            (ts.cosh, 0.5),
            (ts.arcsinh, 0.5),
            (ts.arccosh, 1.5),
            (ts.arctanh, 0.5),
            (ts.sign, 0.5),
            (ts.relu, 0.5),
            (ts.sigmoid, 0.5),
            (ts.tanh, 0.5),
            (ts.softplus, 0.5),
        )

        def evaluate(backend, function, item):
            with ts.use_backend(backend):
                value = ts.Variable(ts.full((64,), item))
                output = function(value)
                gradient = ts.grad(ts.sum(output), value)
                return output.data, gradient

        for function, item in operations:
            with self.subTest(operation=function.__name__):
                expected = evaluate("python", function, item)
                actual = evaluate("numpy", function, item)
                for actual_tensor, expected_tensor in zip(actual, expected):
                    self.assertEqual(actual_tensor.shape, expected_tensor.shape)
                    self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
                    for actual_item, expected_item in zip(
                        actual_tensor._data,
                        expected_tensor._data,
                    ):
                        self.assertAlmostEqual(actual_item, expected_item)

    def test_unary_kernels_preserve_domain_errors(self):
        with ts.use_backend("numpy"):
            with self.assertRaisesRegex(
                ValueError,
                "sqrt is only defined for non-negative values",
            ):
                ts.sqrt(ts.full((64,), -1.0))
            value = ts.Variable(ts.full((64,), 1.0))
            with self.assertRaisesRegex(
                ValueError,
                "arcsin derivative is undefined at -1 and 1",
            ):
                ts.grad(ts.sum(ts.arcsin(value)), value)

    def test_fused_probability_loss_rejects_nonfinite_probabilities(self):
        probabilities = ts.Tensor([float("nan")] * 64)
        targets = ts.full((64,), 0.5)

        with ts.use_backend("numpy"):
            with self.assertRaisesRegex(
                ValueError,
                "probabilities must be between 0 and 1",
            ):
                ts.binary_cross_entropy(probabilities, targets)

    def test_normalization_and_loss_operations_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend,
                "normalization",
                wraps=numpy_backend.normalization,
            ) as normalization,
            patch.object(
                numpy_backend,
                "normalization_gradient",
                wraps=numpy_backend.normalization_gradient,
            ) as normalization_gradient,
            patch.object(
                numpy_backend,
                "logsumexp",
                wraps=numpy_backend.logsumexp,
            ) as logsumexp,
            patch.object(
                numpy_backend,
                "logsumexp_gradient",
                wraps=numpy_backend.logsumexp_gradient,
            ) as logsumexp_gradient,
            patch.object(
                numpy_backend,
                "cross_entropy",
                wraps=numpy_backend.cross_entropy,
            ) as cross_entropy,
            patch.object(
                numpy_backend,
                "cross_entropy_gradient",
                wraps=numpy_backend.cross_entropy_gradient,
            ) as cross_entropy_gradient,
            patch.object(
                numpy_backend,
                "one_hot_targets",
                wraps=numpy_backend.one_hot_targets,
            ) as one_hot_targets,
            patch.object(
                numpy_backend,
                "distributions_valid",
                wraps=numpy_backend.distributions_valid,
            ) as distributions_valid,
            patch.object(
                numpy_backend,
                "binary_cross_entropy",
                wraps=numpy_backend.binary_cross_entropy,
            ) as binary_cross_entropy,
            patch.object(
                numpy_backend,
                "binary_cross_entropy_gradient",
                wraps=numpy_backend.binary_cross_entropy_gradient,
            ) as binary_cross_entropy_gradient,
        ):
            with ts.use_backend("numpy"):
                logits = ts.Variable(ts.full((16, 4), 0.25))
                ts.grad(ts.sum(ts.softmax(logits, axis=1)), logits)
                ts.grad(ts.sum(ts.log_softmax(logits, axis=1)), logits)
                ts.grad(ts.sum(ts.logsumexp(logits, axis=1)), logits)

                classes = ts.Tensor([0] * 16, dtype=ts.int64)
                ts.grad(ts.cross_entropy(logits, classes), logits)

                binary_logits = ts.Variable(ts.full((64,), 0.25))
                binary_targets = ts.full((64,), 0.5)
                ts.grad(
                    ts.binary_cross_entropy(
                        binary_logits,
                        binary_targets,
                        from_logits=True,
                    ),
                    binary_logits,
                )

        self.assertEqual(normalization.call_count, 2)
        self.assertEqual(normalization_gradient.call_count, 2)
        logsumexp.assert_called_once()
        logsumexp_gradient.assert_called_once()
        cross_entropy.assert_called_once()
        cross_entropy_gradient.assert_called_once()
        one_hot_targets.assert_called_once()
        self.assertEqual(distributions_valid.call_count, 2)
        binary_cross_entropy.assert_called_once()
        binary_cross_entropy_gradient.assert_called_once()

    def test_normalization_and_losses_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                logits = ts.Variable(
                    ts.Tensor(
                        [float(index % 4) / 4.0 for index in range(64)],
                        shape=(16, 4),
                    )
                )
                softmax = ts.softmax(logits, axis=1)
                log_softmax = ts.log_softmax(logits, axis=1)
                normalizer = ts.logsumexp(logits, axis=1)
                classes = ts.Tensor([index % 4 for index in range(16)])
                multiclass = ts.cross_entropy(logits, classes)
                multiclass_gradient = ts.grad(multiclass, logits)

                predictions = ts.Variable(
                    ts.Tensor(
                        [0.2 + (index % 5) / 10.0 for index in range(64)]
                    )
                )
                targets = ts.full((64,), 0.5)
                binary = ts.binary_cross_entropy(predictions, targets)
                binary_gradient = ts.grad(binary, predictions)
                return (
                    softmax.data,
                    log_softmax.data,
                    normalizer.data,
                    multiclass.data,
                    multiclass_gradient,
                    binary.data,
                    binary_gradient,
                )

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)

    def test_remaining_reductions_and_selection_dispatch_to_numpy(self):
        value_data = ts.Tensor(
            [1.0 + (index % 8) / 10.0 for index in range(64)],
            shape=(8, 8),
        )
        with (
            patch.object(
                numpy_backend,
                "reduction",
                wraps=numpy_backend.reduction,
            ) as reduction,
            patch.object(
                numpy_backend,
                "reduction_gradient",
                wraps=numpy_backend.reduction_gradient,
            ) as reduction_gradient,
            patch.object(
                numpy_backend,
                "arg_extremum",
                wraps=numpy_backend.arg_extremum,
            ) as arg_extremum,
            patch.object(
                numpy_backend,
                "comparison",
                wraps=numpy_backend.comparison,
            ) as comparison,
            patch.object(
                numpy_backend,
                "where",
                wraps=numpy_backend.where,
            ) as where,
            patch.object(
                numpy_backend,
                "where_gradient",
                wraps=numpy_backend.where_gradient,
            ) as where_gradient,
            patch.object(
                numpy_backend,
                "clip",
                wraps=numpy_backend.clip,
            ) as clip,
            patch.object(
                numpy_backend,
                "clip_gradient",
                wraps=numpy_backend.clip_gradient,
            ) as clip_gradient,
            patch.object(
                numpy_backend,
                "extremum",
                wraps=numpy_backend.extremum,
            ) as extremum,
            patch.object(
                numpy_backend,
                "extremum_gradient",
                wraps=numpy_backend.extremum_gradient,
            ) as extremum_gradient,
        ):
            with ts.use_backend("numpy"):
                for operation in (ts.std, ts.prod, ts.min, ts.max):
                    value = ts.Variable(value_data)
                    output = operation(value, axis=1)
                    ts.grad(output, value, grad_outputs=ts.full((8,), 1.0))

                ts.argmin(value_data, axis=1)
                ts.argmax(value_data, axis=1)

                right = ts.full((8, 8), 1.4)
                for operation in (
                    ts.equal,
                    ts.not_equal,
                    ts.less,
                    ts.less_equal,
                    ts.greater,
                    ts.greater_equal,
                ):
                    operation(value_data, right)

                condition = ts.Tensor(
                    [index % 2 for index in range(64)],
                    dtype=ts.uint8,
                    shape=(8, 8),
                )
                selected = ts.Variable(value_data)
                chosen = ts.where(condition, selected, right)
                ts.grad(chosen, selected, grad_outputs=ts.full((8, 8), 1.0))

                clipped = ts.Variable(value_data)
                clipped_output = ts.clip(clipped, 1.2, 1.6)
                ts.grad(
                    clipped_output,
                    clipped,
                    grad_outputs=ts.full((8, 8), 1.0),
                )

                for operation in (ts.minimum, ts.maximum):
                    selected = ts.Variable(value_data)
                    output = operation(selected, right)
                    ts.grad(
                        output,
                        selected,
                        grad_outputs=ts.full((8, 8), 1.0),
                    )

        self.assertEqual(
            [call.args[0] for call in reduction.call_args_list],
            ["std", "prod", "min", "max"],
        )
        self.assertEqual(
            [call.args[0] for call in reduction_gradient.call_args_list],
            ["std", "prod", "min", "max"],
        )
        self.assertEqual(
            [call.args[0] for call in arg_extremum.call_args_list],
            ["argmin", "argmax"],
        )
        self.assertEqual(comparison.call_count, 6)
        where.assert_called_once()
        where_gradient.assert_called_once()
        clip.assert_called_once()
        clip_gradient.assert_called_once()
        self.assertEqual(extremum.call_count, 2)
        self.assertEqual(extremum_gradient.call_count, 2)

    def test_remaining_reductions_and_selection_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                data = ts.Tensor(
                    [1.0 + (index % 8) / 10.0 for index in range(64)],
                    shape=(8, 8),
                )
                results = []
                for operation in (ts.std, ts.prod, ts.min, ts.max):
                    value = ts.Variable(data)
                    output = operation(value, axis=1)
                    gradient = ts.grad(
                        output,
                        value,
                        grad_outputs=ts.full((8,), 1.0),
                    )
                    results.extend((output.data, gradient))
                results.extend((
                    ts.argmin(data, axis=1),
                    ts.argmax(data, axis=1),
                    ts.greater_equal(data, 1.4),
                    ts.where(ts.greater(data, 1.4), data, 1.4),
                    ts.clip(data, 1.2, 1.6),
                    ts.minimum(data, 1.4),
                    ts.maximum(data, 1.4),
                ))
                return results

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)

    def test_creation_layout_and_optimizers_dispatch_to_numpy(self):
        with (
            patch.object(
                numpy_backend,
                "full",
                wraps=numpy_backend.full,
            ) as full,
            patch.object(
                numpy_backend,
                "eye",
                wraps=numpy_backend.eye,
            ) as eye,
            patch.object(
                numpy_backend,
                "arange",
                wraps=numpy_backend.arange,
            ) as arange,
            patch.object(
                numpy_backend,
                "linspace",
                wraps=numpy_backend.linspace,
            ) as linspace,
            patch.object(
                numpy_backend,
                "transpose",
                wraps=numpy_backend.transpose,
            ) as transpose,
            patch.object(
                numpy_backend,
                "concat",
                wraps=numpy_backend.concat,
            ) as concat,
            patch.object(
                numpy_backend,
                "stack",
                wraps=numpy_backend.stack,
            ) as stack,
            patch.object(
                numpy_backend,
                "outer",
                wraps=numpy_backend.outer,
            ) as outer,
            patch.object(
                numpy_backend,
                "outer_gradient",
                wraps=numpy_backend.outer_gradient,
            ) as outer_gradient,
            patch.object(
                numpy_backend,
                "sgd_update",
                wraps=numpy_backend.sgd_update,
            ) as sgd_update,
            patch.object(
                numpy_backend,
                "adam_update",
                wraps=numpy_backend.adam_update,
            ) as adam_update,
            patch.object(
                numpy_backend,
                "rmsprop_update",
                wraps=numpy_backend.rmsprop_update,
            ) as rmsprop_update,
        ):
            with ts.use_backend("numpy"):
                ts.full((64,), 2.0)
                ts.eye(8)
                ts.arange(64)
                ts.linspace(1.0, 2.0, 64)

                matrix = ts.full((8, 8), 2.0)
                ts.transpose(matrix)
                ts.concat([matrix, matrix], axis=0)
                ts.stack([matrix, matrix], axis=0)

                left = ts.Variable(ts.full((8,), 2.0))
                right = ts.Variable(ts.full((8,), 3.0))
                product = ts.outer(left, right)
                ts.grad(
                    product,
                    (left, right),
                    grad_outputs=ts.full((8, 8), 1.0),
                )

                for optimizer_type in (
                    ts.optim.SGD,
                    ts.optim.Adam,
                    ts.optim.RMSprop,
                ):
                    parameter = ts.Variable(ts.full((64,), 1.0))
                    parameter.grad = ts.full((64,), 1.0)
                    optimizer_type([parameter], learning_rate=0.1).step()

        self.assertGreaterEqual(full.call_count, 1)
        eye.assert_called_once()
        arange.assert_called_once()
        linspace.assert_called_once()
        transpose.assert_called_once()
        concat.assert_called_once()
        stack.assert_called_once()
        outer.assert_called_once()
        outer_gradient.assert_called_once()
        sgd_update.assert_called_once()
        adam_update.assert_called_once()
        rmsprop_update.assert_called_once()

    def test_creation_layout_and_optimizer_results_match_python_backend(self):
        def evaluate(backend):
            with ts.use_backend(backend):
                matrix = ts.reshape(ts.arange(64), (8, 8))
                creations = (
                    ts.full((64,), 1.25),
                    ts.eye(8, k=1),
                    ts.arange(0.0, 6.4, 0.1),
                    ts.linspace(1.0, 2.0, 64),
                )
                layouts = (
                    ts.transpose(matrix),
                    ts.concat([matrix, matrix], axis=1),
                    ts.stack([matrix, matrix], axis=1),
                    ts.outer(ts.ones((8,)), ts.arange(8)),
                )
                parameters = []
                for optimizer_type in (
                    ts.optim.SGD,
                    ts.optim.Adam,
                    ts.optim.RMSprop,
                ):
                    parameter = ts.Variable(ts.full((64,), 1.0))
                    parameter.grad = ts.full((64,), 0.5)
                    optimizer_type([parameter], learning_rate=0.1).step()
                    parameters.append(parameter.data)
                return creations + layouts + tuple(parameters)

        expected = evaluate("python")
        actual = evaluate("numpy")
        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertEqual(actual_tensor.shape, expected_tensor.shape)
            self.assertIs(actual_tensor.dtype, expected_tensor.dtype)
            for actual_item, expected_item in zip(
                actual_tensor._data,
                expected_tensor._data,
            ):
                self.assertAlmostEqual(actual_item, expected_item)

    def test_slice_dispatches_to_numpy(self):
        value = ts.Variable(ts.full((64, 2), 2.0))
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
                value = ts.Variable(ts.full((128,), 2.0))
                output = ts.sum(value[::2])
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

    def test_cast_dispatches_to_numpy(self):
        value = ts.full((64,), 1.25)
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

    def test_broadcast_gradient_reductions_dispatch_to_numpy(self):
        left = ts.Variable(ts.full((64, 1), 2.0))
        right = ts.Variable(ts.full((1, 64), 3.0))
        with (
            patch.object(
                numpy_backend,
                "sum_to_shape",
                wraps=numpy_backend.sum_to_shape,
            ) as sum_kernel,
            patch.object(
                numpy_backend,
                "sum_products_to_shape",
                wraps=numpy_backend.sum_products_to_shape,
            ) as product_kernel,
        ):
            with ts.use_backend("numpy"):
                ts.grad(ts.sum(left + right), (left, right))
                ts.grad(ts.sum(left * right), (left, right))

        self.assertGreaterEqual(sum_kernel.call_count, 2)
        self.assertGreaterEqual(product_kernel.call_count, 2)

    def test_broadcast_gradient_reductions_match_python_backend(self):
        def gradients(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.full((64, 1), 2.0))
                right = ts.Variable(ts.full((1, 64), 3.0))
                added = ts.grad(ts.sum(left + right), (left, right))
                multiplied = ts.grad(ts.sum(left * right), (left, right))
                return (
                    tuple(item.tolist() for item in added),
                    tuple(item.tolist() for item in multiplied),
                )

        self.assertEqual(gradients("numpy"), gradients("python"))

    def test_product_reduction_preserves_exact_cancellation(self):
        def gradient(backend):
            with ts.use_backend(backend):
                left = ts.Variable(ts.Tensor([[1.0], [1.0]]))
                right = ts.Variable(
                    ts.Tensor([[1.0e308, -1.0e308]])
                )
                return ts.grad(ts.sum(left * right), left)

        self.assertEqual(
            gradient("numpy").tolist(),
            gradient("python").tolist(),
        )

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

    def test_dense_target_validation_dispatches_to_numpy(self):
        targets = (
            ts.full((16, 4), 0.25),
            ts.full((16, 4), 0.25),
        )
        targets[0][0, 0] = 0.5
        targets[1][0, 0] = 0.25000015
        with patch.object(
            numpy_backend,
            "distributions_valid",
            wraps=numpy_backend.distributions_valid,
        ) as distributions_valid:
            with ts.use_backend("numpy"):
                for target in targets:
                    with self.assertRaisesRegex(ValueError, "sum to 1"):
                        ts.cross_entropy(ts.full((16, 4), 1.0), target)

        self.assertEqual(distributions_valid.call_count, 2)

    def test_tiny_operations_bypass_numpy_kernel_dispatch(self):
        value = ts.Tensor([2.0])
        with (
            patch.object(numpy_backend, "binary") as binary,
            patch.object(numpy_backend, "negate") as negate,
            patch.object(numpy_backend, "cast_tensor") as cast_tensor,
            patch.object(numpy_backend, "reduction") as reduction,
            patch.object(numpy_backend, "matmul") as matmul,
        ):
            with ts.use_backend("numpy"):
                _ = value + value
                _ = -value
                _ = value.astype(ts.float32)
                _ = ts.sum(value)
                _ = value @ value

        binary.assert_not_called()
        negate.assert_not_called()
        cast_tensor.assert_not_called()
        reduction.assert_not_called()
        matmul.assert_not_called()

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
