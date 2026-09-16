import math
import unittest
from unittest.mock import patch

import tensors as ts


class OptimizerTests(unittest.TestCase):
    def test_sgd_uses_the_optimizer_base_contract(self):
        optimizer = ts.optim.SGD([], learning_rate=0.1)

        self.assertIsInstance(optimizer, ts.optim.Optimizer)

    def test_base_zero_grad_clears_managed_parameter_gradients(self):
        parameter = ts.Variable([1.0])
        parameter.grad = ts.Tensor([2.0])
        optimizer = ts.optim.SGD([parameter], learning_rate=0.1)

        optimizer.zero_grad()

        self.assertIsNone(parameter.grad)

    def test_optimizer_updates_a_duplicate_parameter_once(self):
        parameter = ts.Variable([1.0])
        parameter.grad = ts.Tensor([1.0])
        optimizer = ts.optim.SGD([parameter, parameter], learning_rate=0.1)

        optimizer.step()

        self.assertEqual(optimizer.parameters, (parameter,))
        self.assertEqual(parameter.data.tolist(), [0.9])

    def test_optimizers_reject_tensor_parameters(self):
        for optimizer_type in (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        ):
            with self.subTest(optimizer=optimizer_type.__name__):
                with self.assertRaisesRegex(TypeError, "must be a Variable"):
                    optimizer_type([ts.Tensor([1.0])], learning_rate=0.1)

    def test_optimizers_reject_nonfinite_learning_rates(self):
        for optimizer_type in (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        ):
            for learning_rate in (math.nan, math.inf, -math.inf):
                with self.subTest(
                    optimizer=optimizer_type.__name__,
                    learning_rate=learning_rate,
                ):
                    with self.assertRaises(ValueError):
                        optimizer_type([], learning_rate=learning_rate)

    def test_learning_rate_assignments_remain_validated(self):
        for optimizer_type in (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        ):
            optimizer = optimizer_type([], learning_rate=0.1)
            optimizer.learning_rate = 0.2
            self.assertEqual(optimizer.learning_rate, 0.2)

            for learning_rate in (0.0, math.nan, math.inf, -math.inf):
                with self.subTest(
                    optimizer=optimizer_type.__name__,
                    learning_rate=learning_rate,
                ):
                    with self.assertRaises(ValueError):
                        optimizer.learning_rate = learning_rate
                    self.assertEqual(optimizer.learning_rate, 0.2)

    def test_optimizers_preserve_parameter_dtype(self):
        optimizer_types = (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        )
        for optimizer_type in optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                parameter = ts.Variable(ts.Tensor([1.0], dtype=ts.float32))
                parameter.grad = ts.Tensor([1.0], dtype=ts.float64)
                optimizer = optimizer_type([parameter], learning_rate=0.1)

                optimizer.step()

                self.assertIs(parameter.dtype, ts.float32)

    def test_optimizer_requires_a_step_implementation(self):
        with self.assertRaises(TypeError):
            ts.optim.Optimizer([])


class EveryParameterIsUpdatedTests(unittest.TestCase):
    """Every parameter holding a gradient moves on both execution paths.

    A batched update is an optional optimization: it may decline, and the
    optimizer then updates each parameter individually. Both paths have to
    reach every eligible parameter, so a step that quietly leaves one behind
    is a silent training bug rather than a visible failure.
    """

    OPTIMIZERS = ("SGD", "Adam", "RMSprop")
    BATCHED_ENTRY_POINTS = {
        "SGD": "execute_sgd_updates",
        "Adam": "execute_adam_updates",
        "RMSprop": "execute_rmsprop_updates",
    }

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def _parameters(self, count, size=64):
        parameters = []
        for index in range(count):
            parameter = ts.Variable(ts.full((size,), float(index + 1)))
            parameter.grad = ts.full((size,), 0.5 * (index + 1))
            parameters.append(parameter)
        return parameters

    def _assert_all_moved(self, parameters, before):
        for index, (parameter, original) in enumerate(zip(parameters, before)):
            with self.subTest(parameter=index):
                self.assertNotEqual(
                    parameter.data.tolist(),
                    original,
                    f"parameter {index} was not updated",
                )

    def _step(self, name, parameters):
        optimizer = getattr(ts.optim, name)(parameters, learning_rate=0.1)
        optimizer.step()

    def test_batched_path_updates_every_parameter(self):
        for name in self.OPTIMIZERS:
            for backend in ts.available_backends():
                with self.subTest(optimizer=name, backend=backend):
                    with ts.use_backend(backend):
                        parameters = self._parameters(4)
                        before = [p.data.tolist() for p in parameters]
                        self._step(name, parameters)
                    self._assert_all_moved(parameters, before)

    def test_individual_path_updates_every_parameter(self):
        """Force the batched update to decline, then check each parameter."""
        for name in self.OPTIMIZERS:
            entry_point = self.BATCHED_ENTRY_POINTS[name]
            module = f"tensors.optim.{name.lower()}"
            for backend in ts.available_backends():
                with self.subTest(optimizer=name, backend=backend):
                    with ts.use_backend(backend):
                        parameters = self._parameters(4)
                        before = [p.data.tolist() for p in parameters]
                        with patch(
                            f"{module}.{entry_point}", return_value=None
                        ) as declined:
                            self._step(name, parameters)
                        declined.assert_called()
                    self._assert_all_moved(parameters, before)

    def test_single_parameter_uses_the_individual_path(self):
        for name in self.OPTIMIZERS:
            for backend in ts.available_backends():
                with self.subTest(optimizer=name, backend=backend):
                    with ts.use_backend(backend):
                        parameters = self._parameters(1)
                        before = [p.data.tolist() for p in parameters]
                        self._step(name, parameters)
                    self._assert_all_moved(parameters, before)

    def test_both_paths_reach_the_same_parameter_values(self):
        for name in self.OPTIMIZERS:
            entry_point = self.BATCHED_ENTRY_POINTS[name]
            module = f"tensors.optim.{name.lower()}"
            with self.subTest(optimizer=name):
                batched = self._parameters(4)
                self._step(name, batched)

                individual = self._parameters(4)
                with patch(f"{module}.{entry_point}", return_value=None):
                    self._step(name, individual)

                for index, (left, right) in enumerate(zip(batched, individual)):
                    with self.subTest(parameter=index):
                        for expected, actual in zip(
                            left.data.tolist(), right.data.tolist()
                        ):
                            self.assertAlmostEqual(actual, expected)

    def test_differently_shaped_parameters_are_all_updated(self):
        """A batch splits one result back onto parameters of different sizes."""
        shapes = ((64,), (8, 8), (4, 4, 4), (16,))
        for name in self.OPTIMIZERS:
            for backend in ts.available_backends():
                with self.subTest(optimizer=name, backend=backend):
                    with ts.use_backend(backend):
                        parameters = []
                        for index, shape in enumerate(shapes):
                            parameter = ts.Variable(ts.full(shape, float(index + 1)))
                            parameter.grad = ts.full(shape, 0.5 * (index + 1))
                            parameters.append(parameter)
                        before = [p.data.tolist() for p in parameters]
                        self._step(name, parameters)
                    self._assert_all_moved(parameters, before)
                    for parameter, shape in zip(parameters, shapes):
                        self.assertEqual(parameter.data.shape, shape)

    def test_a_parameter_without_a_gradient_is_left_alone(self):
        for name in self.OPTIMIZERS:
            with self.subTest(optimizer=name):
                parameters = self._parameters(3)
                parameters[1].grad = None
                before = [p.data.tolist() for p in parameters]

                self._step(name, parameters)

                self.assertNotEqual(parameters[0].data.tolist(), before[0])
                self.assertEqual(parameters[1].data.tolist(), before[1])
                self.assertNotEqual(parameters[2].data.tolist(), before[2])


if __name__ == "__main__":
    unittest.main()
