import unittest

import tensors as ts


class AdamTests(unittest.TestCase):
    def test_first_step_uses_bias_corrected_moments(self):
        parameter = ts.Variable([1.0])
        parameter.grad = ts.Tensor([2.0])
        optimizer = ts.optim.Adam([parameter], learning_rate=0.1)

        optimizer.step()

        self.assertAlmostEqual(parameter.data.item(), 0.9)

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

    def test_invalid_hyperparameters_are_rejected(self):
        cases = [
            {"learning_rate": 0.0},
            {"betas": (-0.1, 0.999)},
            {"betas": (0.9, 1.0)},
            {"eps": 0.0},
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    ts.optim.Adam([], **arguments)


if __name__ == "__main__":
    unittest.main()
