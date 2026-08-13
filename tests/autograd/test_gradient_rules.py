import unittest

import tensors as ts


class GradientRuleTests(unittest.TestCase):
    def assertGradient(self, function, inputs, **kwargs):
        self.assertTrue(ts.gradcheck(function, inputs, **kwargs))

    def test_binary_arithmetic_rules(self):
        left = ts.Tensor([[1.2], [2.3]])
        right = ts.Tensor([[0.7, 1.4]])
        cases = {
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / b,
            "power": lambda a, b: a ** b,
        }

        for name, function in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, (left, right))

    def test_scalar_arithmetic_rules(self):
        value = ts.Tensor([0.7, 1.4, 2.1])
        cases = {
            "add": lambda x: x + 2.0,
            "reverse_add": lambda x: 2.0 + x,
            "subtract": lambda x: x - 2.0,
            "reverse_subtract": lambda x: 2.0 - x,
            "multiply": lambda x: x * 2.0,
            "reverse_multiply": lambda x: 2.0 * x,
            "divide": lambda x: x / 2.0,
            "reverse_divide": lambda x: 2.0 / x,
            "power": lambda x: x ** 2.3,
            "reverse_power": lambda x: 2.0 ** x,
            "negate": lambda x: -x,
        }

        for name, function in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, value)

    def test_elementwise_math_rules(self):
        positive = ts.Tensor([0.4, 1.2, 2.5])
        signed = ts.Tensor([-1.2, 0.4, 2.1])
        cases = {
            "sqrt": (lambda x: ts.sqrt(x), positive),
            "exp": (lambda x: ts.exp(x), signed),
            "log": (lambda x: ts.log(x), positive),
            "relu": (lambda x: ts.relu(x), signed),
            "sigmoid": (lambda x: ts.sigmoid(x), signed),
            "tanh": (lambda x: ts.tanh(x), signed),
            "softplus": (lambda x: ts.softplus(x), signed),
        }

        for name, (function, value) in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, value)

    def test_axis_reduction_rules(self):
        value = ts.Tensor([[0.5, 1.5, 3.0], [2.0, 4.0, 7.0]])
        cases = {
            "sum": lambda x: ts.sum(x, axis=1),
            "mean": lambda x: ts.mean(x, axis=0, keepdims=True),
            "min": lambda x: ts.min(x, axis=1),
            "max": lambda x: ts.max(x, axis=0),
            "std": lambda x: ts.std(x, axis=1),
            "norm": lambda x: ts.linalg.norm(x, axis=1),
        }

        for name, function in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, value)

    def test_shape_operation_rules(self):
        left = ts.Tensor([[0.5, 1.5], [2.5, 3.5]])
        right = ts.Tensor([[4.5, 5.5], [6.5, 7.5]])
        cases = {
            "reshape": lambda a, b: ts.reshape(a, (4,)) + ts.reshape(b, (4,)),
            "transpose": lambda a, b: ts.transpose(a) + ts.transpose(b),
            "concat": lambda a, b: ts.concat([a, b], axis=1),
            "stack": lambda a, b: ts.stack([a, b], axis=1),
            "slice": lambda a, b: a[:, 1:] + b[:, :1],
        }

        for name, function in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, (left, right))

    def test_linear_algebra_rules(self):
        matrix = ts.Tensor([[0.5, 1.5], [2.5, 3.5]])
        weights = ts.Tensor([[1.2, 0.7], [0.4, 1.8]])
        vector = ts.Tensor([0.8, 1.4])
        cases = {
            "dot": (lambda a, b: ts.dot(a, b), (matrix, weights)),
            "matmul": (lambda a, b: a @ b, (matrix, weights)),
            "outer": (lambda a, b: ts.outer(a, b), (vector, vector)),
        }

        for name, (function, inputs) in cases.items():
            with self.subTest(operation=name):
                self.assertGradient(function, inputs)

    def test_softmax_rule(self):
        value = ts.Tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])

        self.assertGradient(
            lambda x: ts.softmax(x, axis=1) ** 2.0,
            value,
        )


if __name__ == "__main__":
    unittest.main()
