import math
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
import tensors.backend.cuda as cuda_backend
import tensors.backend.numpy as numpy_backend
from tensors.storage import CudaStorage, NumPyStorage

from ._support import NumPyParityTestCase, requires_cuda, requires_numpy


@requires_cuda
class CudaGroupedOptimizerTests(unittest.TestCase):
    """Grouped optimizer updates keep their fused CUDA path."""

    def test_grouped_adam_remains_accelerated_across_sign_changes(self):
        with ts.use_backend("python"):
            reference_parameters = [
                ts.Variable(ts.full((64,), 1.0)) for _ in range(2)
            ]
            reference_optimizer = ts.optim.Adam(reference_parameters)
            for parameter, gradient in zip(
                reference_parameters,
                (1.0, -1.0),
            ):
                parameter.grad = ts.full((64,), gradient)
            reference_optimizer.step()
            for parameter, gradient in zip(
                reference_parameters,
                (-1.0, 1.0),
            ):
                parameter.grad = ts.full((64,), gradient)
            reference_optimizer.step()
            expected = tuple(
                (
                    parameter.data.tolist(),
                    reference_optimizer._state[id(parameter)]["m"].tolist(),
                    reference_optimizer._state[id(parameter)]["v"].tolist(),
                )
                for parameter in reference_parameters
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

            with patch.object(
                cuda_backend,
                "adam_updates",
                side_effect=observed,
            ):
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
            self.assertIsInstance(parameter.data._storage, CudaStorage)
            self.assertIsInstance(moment._storage, CudaStorage)
            self.assertIsInstance(second_moment._storage, CudaStorage)
            for actual, reference in zip(
                (
                    parameter.data.tolist(),
                    moment.tolist(),
                    second_moment.tolist(),
                ),
                expected_values,
            ):
                for actual_value, expected_value in zip(actual, reference):
                    self.assertTrue(math.isclose(
                        actual_value,
                        expected_value,
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    ))
    def test_grouped_adam_still_rejects_nonfinite_state(self):
        with ts.use_backend("cuda"):
            parameters = tuple(ts.full((64,), 1.0) for _ in range(2))
            gradients = tuple(ts.full((64,), 1.0) for _ in range(2))
            moments = (
                ts.full((64,), float("inf")),
                ts.zeros((64,)),
            )
            scales = tuple(ts.zeros((64,)) for _ in range(2))
            normalized = tuple(ts.zeros((64,)) for _ in range(2))
            result = cuda_backend.adam_updates(
                parameters,
                gradients,
                moments,
                scales,
                normalized,
                beta1=0.9,
                beta2=0.999,
                learning_rate=0.001,
                epsilon=1.0e-8,
                first_corrections=(0.1, 0.1),
                second_corrections=(0.001, 0.001),
            )

        self.assertIsNone(result)
    def test_grouped_optimizers_use_scalar_cuda_validation_flags(self):
        import cupy

        for optimizer_type in (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                with ts.use_backend("cuda"):
                    parameters = [
                        ts.Variable(ts.full((64,), 1.0)) for _ in range(2)
                    ]
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
                        ts.Variable(ts.full(
                            (64,),
                            1.0,
                            dtype=(
                                ts.float32 if index % 2 == 0 else ts.float64
                            ),
                        ))
                        for index in range(4)
                    ]
                    for parameter in reference_parameters:
                        parameter.grad = ts.full(
                            (64,),
                            0.5,
                            dtype=parameter.dtype,
                        )
                    optimizer_type(
                        reference_parameters,
                        learning_rate=0.01,
                    ).step()
                    expected = tuple(
                        parameter.data.tolist()
                        for parameter in reference_parameters
                    )

                with ts.use_backend("cuda"):
                    parameters = [
                        ts.Variable(ts.full(
                            (64,),
                            1.0,
                            dtype=(ts.float32 if index % 2 == 0 else ts.float64),
                        ))
                        for index in range(4)
                    ]
                    for parameter in parameters:
                        parameter.grad = ts.full(
                            (64,),
                            0.5,
                            dtype=parameter.dtype,
                        )
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    grouped = getattr(cuda_backend, grouped_name)
                    single = getattr(cuda_backend, single_name)
                    with (
                        patch.object(
                            cuda_backend,
                            grouped_name,
                            wraps=grouped,
                        ) as grouped_call,
                        patch.object(
                            cuda_backend,
                            single_name,
                            wraps=single,
                        ) as single_call,
                    ):
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()

                grouped_call.assert_called_once()
                single_call.assert_not_called()
                self.assertTrue(all(
                    isinstance(parameter.data._storage, CudaStorage)
                    for parameter in parameters
                ))
                for parameter, expected_values in zip(parameters, expected):
                    for actual, reference in zip(
                        parameter.data.tolist(),
                        expected_values,
                    ):
                        self.assertTrue(math.isclose(
                            actual,
                            reference,
                            rel_tol=1.0e-6,
                            abs_tol=1.0e-7,
                        ))
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
    def test_adam_sign_changes_keep_native_updates_accelerated(self):
        for parameter_count, kernel_name in (
            (1, "adam_update"),
            (2, "adam_updates"),
        ):
            with self.subTest(kernel=kernel_name):
                with ts.use_backend("python"):
                    reference_parameters = [
                        ts.Variable(ts.full((64,), 1.0))
                        for _ in range(parameter_count)
                    ]
                    reference_optimizer = ts.optim.Adam(reference_parameters)
                    first_gradients = tuple(
                        1.0 if index % 2 == 0 else -1.0
                        for index in range(parameter_count)
                    )
                    second_gradients = tuple(
                        -gradient for gradient in first_gradients
                    )
                    for parameter, gradient in zip(
                        reference_parameters,
                        first_gradients,
                    ):
                        parameter.grad = ts.full((64,), gradient)
                    reference_optimizer.step()
                    for parameter, gradient in zip(
                        reference_parameters,
                        second_gradients,
                    ):
                        parameter.grad = ts.full((64,), gradient)
                    reference_optimizer.step()
                    expected = tuple(
                        (
                            parameter.data.tolist(),
                            reference_optimizer._state[id(parameter)][
                                "m"
                            ].tolist(),
                            reference_optimizer._state[id(parameter)][
                                "v"
                            ].tolist(),
                        )
                        for parameter in reference_parameters
                    )

                accelerated_results = []
                with ts.use_backend("numpy"):
                    parameters = [
                        ts.Variable(ts.full((64,), 1.0))
                        for _ in range(parameter_count)
                    ]
                    optimizer = ts.optim.Adam(parameters)
                    for parameter, gradient in zip(
                        parameters,
                        first_gradients,
                    ):
                        parameter.grad = ts.full((64,), gradient)
                    optimizer.step()
                    for parameter, gradient in zip(
                        parameters,
                        second_gradients,
                    ):
                        parameter.grad = ts.full((64,), gradient)

                    kernel = getattr(numpy_backend, kernel_name)

                    def observed(*args, **kwargs):
                        result = kernel(*args, **kwargs)
                        accelerated_results.append(result)
                        return result

                    with patch.object(
                        numpy_backend,
                        kernel_name,
                        side_effect=observed,
                    ):
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
                    self.assertIsInstance(parameter.data._storage, NumPyStorage)
                    self.assertIsInstance(moment._storage, NumPyStorage)
                    self.assertIsInstance(second_moment._storage, NumPyStorage)
                    for actual, reference in zip(
                        (
                            parameter.data.tolist(),
                            moment.tolist(),
                            second_moment.tolist(),
                        ),
                        expected_values,
                    ):
                        for actual_value, expected_value in zip(actual, reference):
                            self.assertTrue(math.isclose(
                                actual_value,
                                expected_value,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-12,
                            ))
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
                    result = numpy_backend.adam_update(
                        *tensors,
                        beta1=0.9,
                        beta2=0.999,
                        learning_rate=0.001,
                        epsilon=1.0e-8,
                        first_correction=0.1,
                        second_correction=0.001,
                    )
                    self.assertIsNone(result)

            overflowing = numpy_backend.adam_update(
                ts.full((64,), 1.0e308),
                ts.full((64,), -1.0),
                zero,
                zero,
                zero,
                beta1=0.9,
                beta2=0.999,
                learning_rate=1.0e308,
                epsilon=1.0e-8,
                first_correction=0.1,
                second_correction=0.001,
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
                numpy,
                "concatenate",
                wraps=numpy.concatenate,
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
                        ts.Variable(ts.full(
                            (64,),
                            1.0,
                            dtype=(
                                ts.float32 if index % 2 == 0 else ts.float64
                            ),
                        ))
                        for index in range(4)
                    ]
                    for parameter in reference_parameters:
                        parameter.grad = ts.full(
                            (64,),
                            0.5,
                            dtype=parameter.dtype,
                        )
                    optimizer_type(
                        reference_parameters,
                        learning_rate=0.01,
                    ).step()
                    expected = tuple(
                        parameter.data.tolist()
                        for parameter in reference_parameters
                    )

                with ts.use_backend("numpy"):
                    parameters = [
                        ts.Variable(ts.full(
                            (64,),
                            1.0,
                            dtype=(ts.float32 if index % 2 == 0 else ts.float64),
                        ))
                        for index in range(4)
                    ]
                    for parameter in parameters:
                        parameter.grad = ts.full(
                            (64,),
                            0.5,
                            dtype=parameter.dtype,
                        )
                    optimizer = optimizer_type(parameters, learning_rate=0.01)
                    grouped = getattr(numpy_backend, grouped_name)
                    single = getattr(numpy_backend, single_name)
                    with (
                        patch.object(
                            numpy_backend,
                            grouped_name,
                            wraps=grouped,
                        ) as grouped_call,
                        patch.object(
                            numpy_backend,
                            single_name,
                            wraps=single,
                        ) as single_call,
                    ):
                        backend_state._clear_backend_kernel_cache()
                        optimizer.step()

                grouped_call.assert_called_once()
                single_call.assert_not_called()
                self.assertTrue(all(
                    isinstance(parameter.data._storage, NumPyStorage)
                    for parameter in parameters
                ))
                for parameter, expected_values in zip(parameters, expected):
                    for actual, reference in zip(
                        parameter.data.tolist(),
                        expected_values,
                    ):
                        self.assertTrue(math.isclose(
                            actual,
                            reference,
                            rel_tol=1.0e-6,
                            abs_tol=1.0e-7,
                        ))


if __name__ == "__main__":
    unittest.main()
