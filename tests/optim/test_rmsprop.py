import math
import unittest

import tensors as ts


class RMSpropTests(unittest.TestCase):
    def test_rmsprop_uses_the_optimizer_base_contract(self):
        optimizer = ts.optim.RMSprop([], learning_rate=0.1)

        self.assertIsInstance(optimizer, ts.optim.Optimizer)

    def test_first_step_uses_the_running_squared_gradient(self):
        parameter = ts.Variable([1.0])
        parameter.grad = ts.Tensor([2.0])
        optimizer = ts.optim.RMSprop([parameter], learning_rate=0.1)

        optimizer.step()

        expected = 1.0 - 0.2 / (math.sqrt(0.04) + 1e-8)
        self.assertAlmostEqual(parameter.data.item(), expected)

    def test_running_squared_gradient_persists_across_steps(self):
        parameter = ts.Variable([1.0])
        optimizer = ts.optim.RMSprop([parameter], learning_rate=0.1, rho=0.5)

        parameter.grad = ts.Tensor([2.0])
        optimizer.step()
        parameter.grad = ts.Tensor([2.0])
        optimizer.step()

        first_update = 0.2 / (math.sqrt(2.0) + 1e-8)
        second_update = 0.2 / (math.sqrt(3.0) + 1e-8)
        self.assertAlmostEqual(parameter.data.item(), 1.0 - first_update - second_update)

    def test_invalid_hyperparameters_are_rejected(self):
        cases = [
            {"learning_rate": 0.0},
            {"rho": -0.1},
            {"rho": 1.0},
            {"eps": 0.0},
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ts.optim.RMSprop([], **arguments)


if __name__ == "__main__":
    unittest.main()
