import math
import pathlib
import unittest
from unittest.mock import patch
import tensors as ts
import tensors.backend as backend_state
import tensors.backend.cuda.kernels as cuda_backend
import tensors.backend.numpy.kernels as numpy_backend
import tensors.backend.python.kernels as python_backend
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.python.storage import PythonStorage
from tests.backend._support import NumPyParityTestCase, requires_cuda, requires_numpy


@requires_cuda
class CudaGroupedOptimizerTests(unittest.TestCase):
    """Grouped optimizer updates keep their fused CUDA path."""

    def test_grouped_adam_remains_accelerated_across_sign_changes(self):
        with ts.use_backend("python"):
            reference_parameters = [ts.Variable(ts.full((64,), 1.0)) for _ in range(2)]
            reference_optimizer = ts.optim.Adam(reference_parameters)
            for parameter, gradient in zip(reference_parameters, (1.0, -1.0)):
                parameter.grad = ts.full((64,), gradient)
            reference_optimizer.step()
            for parameter, gradient in zip(reference_parameters, (-1.0, 1.0)):
                parameter.grad = ts.full((64,), gradient)
            reference_optimizer.step()
            expected = tuple(
                (
                    (
                        parameter.data.tolist(),
                        reference_optimizer._state[id(parameter)]["m"].tolist(),
                        reference_optimizer._state[id(parameter)]["v"].tolist(),
                    )
                    for parameter in reference_parameters
                )
            )
        accelerated_results = []
        with ts.use_backend("cuda"):
            parameters = [ts.Variable(ts.full((64,), 1.0)) for _ in range(2)]
            optimizer = ts.optim.Adam(parameters)
            for parameter, gradient in zip(parameters, (1.0, -1.0)):
                parameter.grad = ts.full((64,), gradient)
            optimizer.step()
            for parameter, gradient in zip(parameters, (-1.0, 1.0)):
                parameter.grad = ts.full((64,), gradient)
            kernel = cuda_backend.adam_updates

            def observed(*args, **kwargs):
                result = kernel(*args, **kwargs)
                accelerated_results.append(result)
                return result

            with patch.object(cuda_backend, "adam_updates", side_effect=observed):
                backend_state._clear_backend_kernel_cache()
                optimizer.step()
        self.assertEqual(len(accelerated_results), 1)
        self.assertIsNotNone(accelerated_results[0])
        for parameter, expected_values in zip(parameters, expected):
            state = optimizer._state[id(parameter)]
            moment = state["m"]
            second_moment = state["v"]
            self.assertIsInstance(moment, ts.Tensor)
            self.assertIsInstance(second_moment, ts.Tensor)
            self.assertIsInstance(parameter.data.backend_storage, CudaStorage)
            self.assertIsInstance(moment.backend_storage, CudaStorage)
            self.assertIsInstance(second_moment.backend_storage, CudaStorage)
            for actual, reference in zip(
                (parameter.data.tolist(), moment.tolist(), second_moment.tolist()),
                expected_values,
            ):
                for actual_value, expected_value in zip(actual, reference):
                    self.assertTrue(
                        math.isclose(
                            actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12
                        )
                    )

    def test_grouped_adam_still_rejects_nonfinite_state(self):
        with ts.use_backend("cuda"):
            parameters = tuple((ts.full((64,), 1.0) for _ in range(2)))
            gradients = tuple((ts.full((64,), 1.0) for _ in range(2)))
            moments = (ts.full((64,), float("inf")), ts.zeros((64,)))
            scales = tuple((ts.zeros((64,)) for _ in range(2)))
            normalized = tuple((ts.zeros((64,)) for _ in range(2)))
            tensors = (parameters, gradients, moments, scales, normalized)
            native = tuple(
                tuple(
                    value.backend_storage.buffer.reshape(value.shape) for value in group
                )
                for group in tensors
            )
            result = cuda_backend.adam_updates(
                *native,
                beta1=0.9,
                beta2=0.999,
                learning_rate=0.001,
                epsilon=1e-08,
                first_corrections=(0.1, 0.1),
                second_corrections=(0.001, 0.001),
                dtypes=(ts.float64, ts.float64),
                shapes=((64,), (64,)),
            )
        self.assertIsNone(result)

    def test_grouped_optimizers_use_scalar_cuda_validation_flags(self):
        import cupy

        for optimizer_type in (ts.optim.SGD, ts.optim.Adam, ts.optim.RMSprop):
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("cuda"):
                    parameters = [ts.Variable(ts.full((64,), 1.0)) for _ in range(2)]
                    for parameter in parameters:
                        parameter.grad = ts.full((64,), 0.5)
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    with patch.object(cupy, "any", wraps=cupy.any) as any_call:
                        optimizer.step()
                any_call.assert_not_called()

    def test_mixed_dtype_optimizers_use_grouped_cuda_updates(self):
        optimizers = (
            (ts.optim.SGD, "sgd_updates", "sgd_update"),
            (ts.optim.Adam, "adam_updates", "adam_update"),
            (ts.optim.RMSprop, "rmsprop_updates", "rmsprop_update"),
        )
        for optimizer_type, grouped_name, single_name in optimizers:
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("python"):
                    reference_parameters = [
                        ts.Variable(
                            ts.full(
                                (64,),
                                1.0,
                                dtype=ts.float32 if index % 2 == 0 else ts.float64,
                            )
                        )
                        for index in range(4)
                    ]
                    for parameter in reference_parameters:
                        parameter.grad = ts.full((64,), 0.5, dtype=parameter.dtype)
                    optimizer_type(reference_parameters, learning_rate=0.01).step()
                    expected = tuple(
                        (parameter.data.tolist() for parameter in reference_parameters)
                    )
                with ts.use_backend("cuda"):
                    parameters = [
                        ts.Variable(
                            ts.full(
                                (64,),
                                1.0,
                                dtype=ts.float32 if index % 2 == 0 else ts.float64,
                            )
                        )
                        for index in range(4)
                    ]
                    for parameter in parameters:
                        parameter.grad = ts.full((64,), 0.5, dtype=parameter.dtype)
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    grouped = getattr(cuda_backend, grouped_name)
                    single = getattr(cuda_backend, single_name)
                    with (
                        patch.object(
                            cuda_backend, grouped_name, wraps=grouped
                        ) as grouped_call,
                        patch.object(
                            cuda_backend, single_name, wraps=single
                        ) as single_call,
                    ):
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()
                grouped_call.assert_called_once()
                single_call.assert_not_called()
                self.assertTrue(
                    all(
                        (
                            isinstance(parameter.data.backend_storage, CudaStorage)
                            for parameter in parameters
                        )
                    )
                )
                for parameter, expected_values in zip(parameters, expected):
                    for actual, reference in zip(
                        parameter.data.tolist(), expected_values
                    ):
                        self.assertTrue(
                            math.isclose(
                                actual, reference, rel_tol=1e-06, abs_tol=1e-07
                            )
                        )

    def test_multi_parameter_sgd_uses_grouped_cuda_update(self):
        with ts.use_backend("cuda"):
            parameters = [
                ts.Variable(ts.full((64,), float(index + 1))) for index in range(4)
            ]
            for parameter in parameters:
                parameter.grad = ts.full((64,), 0.5)
            optimizer = ts.optim.SGD(parameters, learning_rate=0.1)
            with patch.object(
                cuda_backend, "sgd_updates", wraps=cuda_backend.sgd_updates
            ) as grouped:
                optimizer.step()
        grouped.assert_called_once()
        self.assertTrue(
            all(
                (
                    isinstance(parameter.data.backend_storage, CudaStorage)
                    for parameter in parameters
                )
            )
        )


@requires_numpy
class NumPyGroupedOptimizerTests(NumPyParityTestCase):
    """Batched optimizer kernels dispatch and match the reference."""

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
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    kernel = getattr(numpy_backend, kernel_name)
                    with patch.object(
                        numpy_backend, kernel_name, wraps=kernel
                    ) as grouped:
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()
                grouped.assert_called_once()
                self.assertTrue(
                    all(
                        (
                            isinstance(parameter.data.backend_storage, NumPyStorage)
                            for parameter in parameters
                        )
                    )
                )

    def test_grouped_optimizer_updates_match_python(self):

        def updated(backend, optimizer_type):
            with ts.use_backend(backend):
                parameters = [
                    ts.Variable(ts.full((64,), float(index + 1))) for index in range(4)
                ]
                for index, parameter in enumerate(parameters):
                    parameter.grad = ts.full((64,), 0.25 * (index + 1))
                optimizer = optimizer_type(parameters, learning_rate=0.01)
                optimizer.step()
                optimizer.step()
                return tuple((parameter.data.tolist() for parameter in parameters))

        for optimizer_type in (ts.optim.SGD, ts.optim.Adam, ts.optim.RMSprop):
            with self.subTest(optimizer=optimizer_type.__name__):
                expected = updated("python", optimizer_type)
                actual = updated("numpy", optimizer_type)
                for expected_values, actual_values in zip(expected, actual):
                    for expected_value, actual_value in zip(
                        expected_values, actual_values
                    ):
                        self.assertAlmostEqual(actual_value, expected_value)

    def test_adam_sign_changes_keep_native_updates_accelerated(self):
        for parameter_count, kernel_name in ((1, "adam_update"), (2, "adam_updates")):
            with self.subTest(kernel=kernel_name):
                with ts.use_backend("python"):
                    reference_parameters = [
                        ts.Variable(ts.full((64,), 1.0)) for _ in range(parameter_count)
                    ]
                    reference_optimizer = ts.optim.Adam(reference_parameters)
                    first_gradients = tuple(
                        (
                            1.0 if index % 2 == 0 else -1.0
                            for index in range(parameter_count)
                        )
                    )
                    second_gradients = tuple(
                        (-gradient for gradient in first_gradients)
                    )
                    for parameter, gradient in zip(
                        reference_parameters, first_gradients
                    ):
                        parameter.grad = ts.full((64,), gradient)
                    reference_optimizer.step()
                    for parameter, gradient in zip(
                        reference_parameters, second_gradients
                    ):
                        parameter.grad = ts.full((64,), gradient)
                    reference_optimizer.step()
                    expected = tuple(
                        (
                            (
                                parameter.data.tolist(),
                                reference_optimizer._state[id(parameter)]["m"].tolist(),
                                reference_optimizer._state[id(parameter)]["v"].tolist(),
                            )
                            for parameter in reference_parameters
                        )
                    )
                accelerated_results = []
                with ts.use_backend("numpy"):
                    parameters = [
                        ts.Variable(ts.full((64,), 1.0)) for _ in range(parameter_count)
                    ]
                    optimizer = ts.optim.Adam(parameters)
                    for parameter, gradient in zip(parameters, first_gradients):
                        parameter.grad = ts.full((64,), gradient)
                    optimizer.step()
                    for parameter, gradient in zip(parameters, second_gradients):
                        parameter.grad = ts.full((64,), gradient)
                    kernel = getattr(numpy_backend, kernel_name)

                    def observed(*args, **kwargs):
                        result = kernel(*args, **kwargs)
                        accelerated_results.append(result)
                        return result

                    with patch.object(numpy_backend, kernel_name, side_effect=observed):
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()
                self.assertEqual(len(accelerated_results), 1)
                self.assertIsNotNone(accelerated_results[0])
                for parameter, expected_values in zip(parameters, expected):
                    state = optimizer._state[id(parameter)]
                    moment = state["m"]
                    second_moment = state["v"]
                    self.assertIsInstance(moment, ts.Tensor)
                    self.assertIsInstance(second_moment, ts.Tensor)
                    self.assertIsInstance(parameter.data.backend_storage, NumPyStorage)
                    self.assertIsInstance(moment.backend_storage, NumPyStorage)
                    self.assertIsInstance(second_moment.backend_storage, NumPyStorage)
                    for actual, reference in zip(
                        (
                            parameter.data.tolist(),
                            moment.tolist(),
                            second_moment.tolist(),
                        ),
                        expected_values,
                    ):
                        for actual_value, expected_value in zip(actual, reference):
                            self.assertTrue(
                                math.isclose(
                                    actual_value,
                                    expected_value,
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                            )

    def test_adam_acceleration_still_rejects_nonfinite_values(self):
        with ts.use_backend("numpy"):
            finite = ts.full((64,), 1.0)
            zero = ts.zeros((64,))
            nonfinite = ts.full((64,), float("inf"))
            cases = (
                (nonfinite, finite, zero, zero, zero),
                (finite, nonfinite, zero, zero, zero),
                (finite, finite, nonfinite, zero, zero),
                (finite, finite, zero, nonfinite, zero),
                (finite, finite, zero, zero, nonfinite),
            )
            for index, tensors in enumerate(cases):
                with self.subTest(nonfinite_input=index):
                    native = tuple(
                        value.backend_storage.buffer.reshape(value.shape)
                        for value in tensors
                    )
                    result = numpy_backend.adam_update(
                        *native,
                        beta1=0.9,
                        beta2=0.999,
                        learning_rate=0.001,
                        epsilon=1e-08,
                        first_correction=0.1,
                        second_correction=0.001,
                        dtype=ts.float64,
                        shape=(64,),
                    )
                    self.assertIsNone(result)
            parameter = ts.full((64,), 1e308)
            gradient = ts.full((64,), -1.0)
            overflowing = numpy_backend.adam_update(
                parameter.backend_storage.buffer.reshape(parameter.shape),
                gradient.backend_storage.buffer.reshape(gradient.shape),
                *(zero.backend_storage.buffer.reshape(zero.shape) for _ in range(3)),
                beta1=0.9,
                beta2=0.999,
                learning_rate=1e308,
                epsilon=1e-08,
                first_correction=0.1,
                second_correction=0.001,
                dtype=ts.float64,
                shape=(64,),
            )
        self.assertIsNone(overflowing)

    def test_grouped_optimizer_packing_reuses_native_buffers(self):
        import numpy

        with ts.use_backend("numpy"):
            parameters = [ts.Variable(ts.full((64,), 1.0)) for _ in range(4)]
            for parameter in parameters:
                parameter.grad = ts.full((64,), 0.5)
            optimizer = ts.optim.SGD(parameters, learning_rate=0.01)
            with patch.object(
                numpy, "concatenate", wraps=numpy.concatenate
            ) as concatenate:
                optimizer.step()
                optimizer.step()
        self.assertEqual(concatenate.call_count, 4)
        outputs = [call.kwargs["out"] for call in concatenate.call_args_list]
        self.assertIs(outputs[0], outputs[2])
        self.assertIs(outputs[1], outputs[3])

    def test_mixed_dtype_optimizers_use_grouped_numpy_updates(self):
        optimizers = (
            (ts.optim.SGD, "sgd_updates", "sgd_update"),
            (ts.optim.Adam, "adam_updates", "adam_update"),
            (ts.optim.RMSprop, "rmsprop_updates", "rmsprop_update"),
        )
        for optimizer_type, grouped_name, single_name in optimizers:
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("python"):
                    reference_parameters = [
                        ts.Variable(
                            ts.full(
                                (64,),
                                1.0,
                                dtype=ts.float32 if index % 2 == 0 else ts.float64,
                            )
                        )
                        for index in range(4)
                    ]
                    for parameter in reference_parameters:
                        parameter.grad = ts.full((64,), 0.5, dtype=parameter.dtype)
                    optimizer_type(reference_parameters, learning_rate=0.01).step()
                    expected = tuple(
                        parameter.data.tolist() for parameter in reference_parameters
                    )
                with ts.use_backend("numpy"):
                    parameters = [
                        ts.Variable(
                            ts.full(
                                (64,),
                                1.0,
                                dtype=ts.float32 if index % 2 == 0 else ts.float64,
                            )
                        )
                        for index in range(4)
                    ]
                    for parameter in parameters:
                        parameter.grad = ts.full((64,), 0.5, dtype=parameter.dtype)
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    grouped = getattr(numpy_backend, grouped_name)
                    single = getattr(numpy_backend, single_name)
                    with (
                        patch.object(
                            numpy_backend, grouped_name, wraps=grouped
                        ) as grouped_call,
                        patch.object(
                            numpy_backend, single_name, wraps=single
                        ) as single_call,
                    ):
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()
                grouped_call.assert_called_once()
                single_call.assert_not_called()
                self.assertTrue(
                    all(
                        isinstance(parameter.data.backend_storage, NumPyStorage)
                        for parameter in parameters
                    )
                )
                for parameter, expected_values in zip(parameters, expected):
                    for actual, reference in zip(
                        parameter.data.tolist(), expected_values
                    ):
                        self.assertTrue(
                            math.isclose(
                                actual, reference, rel_tol=1e-06, abs_tol=1e-07
                            )
                        )


class OptimizerSelectedBackendTests(unittest.TestCase):
    def _parameter(self, size, *, dtype=ts.float64):
        parameter = ts.Variable(ts.full((size,), 1.0, dtype=dtype))
        parameter.grad = ts.full((size,), 0.5, dtype=dtype)
        return parameter

    def test_small_and_large_updates_remain_on_the_selected_backend(self):
        for backend in ts.available_backends():
            for optimizer_type in (ts.optim.SGD, ts.optim.Adam, ts.optim.RMSprop):
                for size in (1, 64):
                    with self.subTest(
                        backend=backend, optimizer=optimizer_type.__name__, size=size
                    ):
                        with ts.use_backend(backend):
                            parameter = self._parameter(size)
                            optimizer = optimizer_type([parameter], learning_rate=0.01)
                            optimizer.step()
                            self.assertEqual(
                                parameter.data.backend_storage.kind, backend
                            )
                            if isinstance(optimizer, ts.optim.Adam):
                                state = optimizer._state[id(parameter)]
                                tensors = tuple(
                                    value
                                    for value in state.values()
                                    if isinstance(value, ts.Tensor)
                                )
                                self.assertTrue(tensors)
                                self.assertTrue(
                                    all(
                                        value.backend_storage.kind == backend
                                        for value in tensors
                                    )
                                )
                            elif isinstance(optimizer, ts.optim.RMSprop):
                                self.assertTrue(
                                    all(
                                        value.backend_storage.kind == backend
                                        for value in optimizer._scaled_state[
                                            id(parameter)
                                        ]
                                    )
                                )

    def test_python_individual_kernels_run_at_every_size(self):
        names = {
            ts.optim.SGD: "sgd_update",
            ts.optim.Adam: "adam_update",
            ts.optim.RMSprop: "rmsprop_update",
        }
        with ts.use_backend("python"):
            for optimizer_type, kernel_name in names.items():
                for size in (1, 64):
                    with self.subTest(optimizer=optimizer_type.__name__, size=size):
                        parameter = self._parameter(size)
                        kernel = getattr(python_backend, kernel_name)
                        with patch.object(
                            python_backend, kernel_name, wraps=kernel
                        ) as called:
                            backend_state._clear_backend_kernel_cache()
                            optimizer_type([parameter], learning_rate=0.01).step()
                        called.assert_called_once()
                        self.assertTrue(
                            all(
                                not isinstance(value, ts.Tensor)
                                for value in called.call_args.args
                            )
                        )

    def test_python_sgd_preserves_intermediate_float32_rounding(self):
        with ts.use_backend("python"):
            parameter = ts.Variable(ts.Tensor([0.006409141700714827], dtype=ts.float32))
            parameter.grad = ts.Tensor([-103091.53125], dtype=ts.float32)
            ts.optim.SGD([parameter], learning_rate=2.840708429328158).step()
        self.assertEqual(parameter.data.tolist(), [292852.96875])

    def test_optimizer_kernel_sources_are_tensor_free(self):
        roots = (
            pathlib.Path("tensors/backend/python/kernels/optim"),
            pathlib.Path("tensors/backend/numpy/kernels/optim"),
            pathlib.Path("tensors/backend/cuda/kernels/optim"),
        )
        forbidden = (
            "Tensor",
            ".get_host_values(",
            "tensor_to_logical_array",
            "_working_values",
            ".tolist(",
            "asnumpy",
            ".get()",
            "get_backend",
            "load_backend",
            "_backend_kernel",
        )
        for root in roots:
            for path in root.glob("*.py"):
                source = path.read_text(encoding="utf-8")
                for token in forbidden:
                    with self.subTest(path=str(path), token=token):
                        self.assertNotIn(token, source)


@requires_numpy
class NumPyOptimizerContractTests(unittest.TestCase):
    def _parameter(self, size=2):
        parameter = ts.Variable(ts.full((size,), 1.0))
        parameter.grad = ts.full((size,), 0.5)
        return parameter

    def test_same_dtype_foreign_gradient_is_rejected_before_state_initialization(self):
        with ts.use_backend("numpy"):
            parameter = ts.Variable(ts.full((2,), 1.0))
            optimizer = ts.optim.Adam([parameter])
        with ts.use_backend("python"):
            parameter.grad = ts.full((2,), 0.5)
        with ts.use_backend("numpy"):
            with self.assertRaises(ts.BackendMismatchError):
                optimizer.step()
        self.assertEqual(optimizer._state, {})

    def test_foreign_parameter_is_rejected_before_state_initialization(self):
        with ts.use_backend("python"):
            parameter = self._parameter()
            optimizer = ts.optim.RMSprop([parameter])
        with ts.use_backend("numpy"):
            with self.assertRaises(ts.BackendMismatchError):
                optimizer.step()
        self.assertEqual(optimizer._scaled_state, {})

    def test_matching_metadata_does_not_hide_foreign_state(self):
        for optimizer_type in (ts.optim.Adam, ts.optim.RMSprop):
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("python"):
                    parameter = self._parameter()
                    optimizer = optimizer_type([parameter])
                    optimizer.step()
                with ts.use_backend("numpy"):
                    parameter.data = ts.full((2,), 1.0)
                    parameter.grad = ts.full((2,), 0.5)
                    with self.assertRaises(ts.BackendMismatchError):
                        optimizer.step()

    def test_same_backend_metadata_change_resets_state(self):
        for optimizer_type in (ts.optim.Adam, ts.optim.RMSprop):
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("numpy"):
                    parameter = self._parameter()
                    optimizer = optimizer_type([parameter])
                    optimizer.step()
                    parameter.data = ts.full((3,), 1.0, dtype=ts.float32)
                    parameter.grad = ts.full((3,), 0.5, dtype=ts.float32)
                    optimizer.step()
                    if isinstance(optimizer, ts.optim.Adam):
                        state_values = tuple(
                            value
                            for value in optimizer._state[id(parameter)].values()
                            if isinstance(value, ts.Tensor)
                        )
                    else:
                        state_values = optimizer._scaled_state[id(parameter)]
                self.assertTrue(all(value.shape == (3,) for value in state_values))
                self.assertTrue(
                    all(value.dtype is ts.float32 for value in state_values)
                )
                self.assertTrue(
                    all(value.backend_storage.kind == "numpy" for value in state_values)
                )

    def test_adam_legacy_visible_state_is_reconstructed_on_selected_backend(self):
        with ts.use_backend("numpy"):
            parameter = self._parameter()
            optimizer = ts.optim.Adam([parameter])
            optimizer._state[id(parameter)] = {
                "step": 1,
                "m": ts.zeros((2,)),
                "v": ts.full((2,), 0.25),
                "beta1_product": 0.9,
                "beta2_product": 0.999,
            }
            optimizer.step()
            state = optimizer._state[id(parameter)]
        self.assertEqual(state["v_scale"].backend_storage.kind, "numpy")
        self.assertEqual(state["v_scaled"].backend_storage.kind, "numpy")

    def test_wrong_backend_result_is_rejected_without_commit(self):
        with ts.use_backend("numpy"):
            parameter = self._parameter(1)
            before = parameter.data
            with patch.object(
                numpy_backend,
                "sgd_update",
                return_value=PythonStorage.from_values([0.0], ts.float64),
            ):
                backend_state._clear_backend_kernel_cache()
                with self.assertRaises(ts.BackendMismatchError):
                    ts.optim.SGD([parameter], learning_rate=0.1).step()
        self.assertIs(parameter.data, before)

    def test_individual_decline_is_unsupported_and_does_not_commit(self):
        with ts.use_backend("numpy"):
            parameter = self._parameter(1)
            before = parameter.data
            with patch.object(numpy_backend, "sgd_update", return_value=None):
                backend_state._clear_backend_kernel_cache()
                with self.assertRaises(ts.BackendOperationUnsupportedError):
                    ts.optim.SGD([parameter], learning_rate=0.1).step()
        self.assertIs(parameter.data, before)

    def test_grouped_decline_retries_only_numpy_individual_kernels(self):
        with ts.use_backend("numpy"):
            parameters = [self._parameter() for _ in range(2)]
            single = numpy_backend.sgd_update
            with (
                patch.object(
                    numpy_backend, "sgd_updates", return_value=None
                ) as grouped,
                patch.object(numpy_backend, "sgd_update", wraps=single) as individual,
                patch.object(python_backend, "sgd_update") as python,
            ):
                backend_state._clear_backend_kernel_cache()
                ts.optim.SGD(parameters, learning_rate=0.1).step()
        grouped.assert_called_once()
        self.assertEqual(individual.call_count, 2)
        python.assert_not_called()

    def test_grouped_decline_then_individual_failure_is_fully_atomic(self):
        with ts.use_backend("numpy"):
            parameters = [self._parameter() for _ in range(2)]
            optimizer = ts.optim.Adam(parameters)
            before = tuple(parameter.data for parameter in parameters)
            individual = numpy_backend.adam_update
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                return individual(*args, **kwargs) if calls == 1 else None

            with (
                patch.object(numpy_backend, "adam_updates", return_value=None),
                patch.object(numpy_backend, "adam_update", side_effect=fail_second),
            ):
                backend_state._clear_backend_kernel_cache()
                with self.assertRaises(ts.BackendOperationUnsupportedError):
                    optimizer.step()
        self.assertEqual(optimizer._state, {})
        self.assertTrue(
            all(
                parameter.data is original
                for parameter, original in zip(parameters, before)
            )
        )


if __name__ == "__main__":
    unittest.main()
