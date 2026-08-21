import math
import unittest

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

    def test_optimizers_preserve_parameter_dtype(self):
        optimizer_types = (
            ts.optim.SGD,
            ts.optim.Adam,
            ts.optim.RMSprop,
        )
        for optimizer_type in optimizer_types:
            with self.subTest(optimizer=optimizer_type.__name__):
                parameter = ts.Variable(
                    ts.Tensor([1.0], dtype=ts.float32)
                )
                parameter.grad = ts.Tensor([1.0], dtype=ts.float64)
                optimizer = optimizer_type([parameter], learning_rate=0.1)

                optimizer.step()

                self.assertIs(parameter.dtype, ts.float32)

    def test_optimizer_requires_a_step_implementation(self):
        with self.assertRaises(TypeError):
            ts.optim.Optimizer([])


if __name__ == "__main__":
    unittest.main()
