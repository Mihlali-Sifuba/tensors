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

    def test_state_resets_when_parameter_metadata_changes(self):
        replacements = (
            ts.Tensor([5.0]),
            ts.Tensor([5.0, 6.0], dtype=ts.float32),
        )
        for replacement in replacements:
            with self.subTest(
                shape=replacement.shape,
                dtype=replacement.dtype.name,
            ):
                parameter = ts.Variable([1.0, 2.0])
                optimizer = ts.optim.RMSprop([parameter], learning_rate=0.1)
                parameter.grad = ts.Tensor([1.0, 1.0])
                optimizer.step()

                initial_values = replacement.tolist()
                parameter.data = replacement
                parameter.grad = ts.Tensor(
                    [1.0] * replacement.size,
                    dtype=replacement.dtype,
                    shape=replacement.shape,
                )
                optimizer.step()

                state = optimizer._state[id(parameter)]
                self.assertEqual(state.shape, replacement.shape)
                self.assertIs(state.dtype, replacement.dtype)
                for value in state.tolist():
                    self.assertAlmostEqual(value, 0.01)
                update = 0.1 / (math.sqrt(0.01) + 1e-8)
                for actual, initial in zip(
                    parameter.data.tolist(),
                    initial_values,
                ):
                    self.assertAlmostEqual(actual, initial - update, places=6)

    def test_invalid_hyperparameters_are_rejected(self):
        cases = [
            {"learning_rate": 0.0},
            {"rho": -0.1},
            {"rho": 1.0},
            {"rho": math.nan},
            {"eps": 0.0},
            {"eps": math.nan},
            {"eps": math.inf},
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ts.optim.RMSprop([], **arguments)

    def test_hyperparameter_assignments_remain_validated(self):
        optimizer = ts.optim.RMSprop([])
        cases = (
            ("rho", math.nan),
            ("rho", math.inf),
            ("rho", 1.0),
            ("eps", math.nan),
            ("eps", math.inf),
            ("eps", 0.0),
        )

        for attribute, value in cases:
            with self.subTest(attribute=attribute, value=value):
                previous = getattr(optimizer, attribute)
                with self.assertRaises(ValueError):
                    setattr(optimizer, attribute, value)
                self.assertEqual(getattr(optimizer, attribute), previous)


if __name__ == "__main__":
    unittest.main()
