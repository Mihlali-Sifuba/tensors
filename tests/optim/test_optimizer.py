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

    def test_optimizer_requires_a_step_implementation(self):
        with self.assertRaises(TypeError):
            ts.optim.Optimizer([])


if __name__ == "__main__":
    unittest.main()
