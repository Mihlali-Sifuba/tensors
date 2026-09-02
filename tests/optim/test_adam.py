import math
import unittest

import tensors as ts


class AdamTests(unittest.TestCase):
    def test_first_step_uses_bias_corrected_moments(self):
        parameter = ts.Variable([1.0])
        parameter.grad = ts.Tensor([2.0])
        optimizer = ts.optim.Adam([parameter], learning_rate=0.1)

        optimizer.step()

        self.assertAlmostEqual(parameter.data.item(), 0.9)

    def test_moment_updates_allow_gradient_sign_changes(self):
        beta1 = 0.9
        beta2 = 0.999
        learning_rate = 0.1
        epsilon = 1.0e-8
        for first_gradient, second_gradient in ((2.0, -1.0), (-2.0, 1.0)):
            with self.subTest(
                first_gradient=first_gradient,
                second_gradient=second_gradient,
            ):
                parameter_value = 1.0
                first_moment = (1.0 - beta1) * first_gradient
                second_moment = (1.0 - beta2) * first_gradient * first_gradient
                parameter_value -= learning_rate * (
                    first_moment / (1.0 - beta1)
                ) / (
                    math.sqrt(second_moment / (1.0 - beta2)) + epsilon
                )
                first_moment = (
                    beta1 * first_moment
                    + (1.0 - beta1) * second_gradient
                )
                second_moment = (
                    beta2 * second_moment
                    + (1.0 - beta2) * second_gradient * second_gradient
                )
                parameter_value -= learning_rate * (
                    first_moment / (1.0 - beta1**2)
                ) / (
                    math.sqrt(second_moment / (1.0 - beta2**2)) + epsilon
                )

                with ts.use_backend("python"):
                    parameter = ts.Variable([1.0])
                    optimizer = ts.optim.Adam(
                        [parameter],
                        learning_rate=learning_rate,
                        betas=(beta1, beta2),
                        eps=epsilon,
                    )
                    parameter.grad = ts.Tensor([first_gradient])
                    optimizer.step()
                    parameter.grad = ts.Tensor([second_gradient])
                    optimizer.step()

                state = optimizer._state[id(parameter)]
                actual_moment = state["m"]
                actual_second_moment = state["v"]
                self.assertIsInstance(actual_moment, ts.Tensor)
                self.assertIsInstance(actual_second_moment, ts.Tensor)
                self.assertTrue(math.isclose(
                    parameter.data.item(),
                    parameter_value,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ))
                self.assertTrue(math.isclose(
                    actual_moment.item(),
                    first_moment,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ))
                self.assertTrue(math.isclose(
                    actual_second_moment.item(),
                    second_moment,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ))

    def test_parameter_uses_its_own_first_gradient_step(self):
        first = ts.Variable([1.0])
        second = ts.Variable([1.0])
        optimizer = ts.optim.Adam([first, second], learning_rate=0.1)

        first.grad = ts.Tensor([2.0])
        optimizer.step()
        first.grad = None
        second.grad = ts.Tensor([2.0])
        optimizer.step()

        self.assertAlmostEqual(second.data.item(), 0.9)

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
                optimizer = ts.optim.Adam([parameter], learning_rate=0.1)
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
                first_moment = state["m"]
                second_moment = state["v"]
                self.assertEqual(state["step"], 1)
                self.assertIsInstance(first_moment, ts.Tensor)
                self.assertIsInstance(second_moment, ts.Tensor)
                self.assertEqual(first_moment.shape, replacement.shape)
                self.assertEqual(second_moment.shape, replacement.shape)
                self.assertIs(first_moment.dtype, replacement.dtype)
                self.assertIs(second_moment.dtype, replacement.dtype)
                for actual, initial in zip(
                    parameter.data.tolist(),
                    initial_values,
                ):
                    self.assertAlmostEqual(actual, initial - 0.1, places=6)

    def test_invalid_hyperparameters_are_rejected(self):
        cases = [
            {"learning_rate": 0.0},
            {"betas": (-0.1, 0.999)},
            {"betas": (0.9, 1.0)},
            {"betas": (math.nan, 0.999)},
            {"eps": 0.0},
            {"eps": math.nan},
            {"eps": math.inf},
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ts.optim.Adam([], **arguments)

    def test_hyperparameter_assignments_remain_validated(self):
        optimizer = ts.optim.Adam([])
        cases = (
            ("beta1", math.nan),
            ("beta1", 1.0),
            ("beta2", math.inf),
            ("beta2", -0.1),
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
