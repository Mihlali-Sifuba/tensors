import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariableMathOperationTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_elementwise_math_functions_return_variables_with_operation_nodes(self):
        variable = ts.Variable([1.0])

        results = {
            "exp": ts.exp(variable),
            "log": ts.log(variable),
            "sqrt": ts.sqrt(variable),
            "relu": ts.relu(variable),
            "sigmoid": ts.sigmoid(variable),
            "tanh": ts.tanh(variable),
            "softplus": ts.softplus(variable),
        }

        for label, result in results.items():
            with self.subTest(operation=label):
                self.assertIsInstance(result, ts.Variable)
                self.assertEqual(result.node.producer.label, label)
                self.assertEqual(result.node.producer.inputs, [variable.node])

    def test_log_of_exp_has_the_original_value_and_unit_gradient(self):
        variable = ts.Variable([0.5, 1.5])
        result = ts.log(ts.exp(variable))

        ts.backward(ts.sum(result))

        self.assertEqual(result.data.tolist(), [0.5, 1.5])
        self.assertEqual(variable.grad.tolist(), [1.0, 1.0])

    def test_relu_only_propagates_gradient_through_positive_values(self):
        variable = ts.Variable([-2.0, 0.0, 3.0])
        result = ts.relu(variable)

        ts.backward(ts.sum(result))

        self.assertEqual(result.data.tolist(), [0.0, 0.0, 3.0])
        self.assertEqual(variable.grad.tolist(), [0.0, 0.0, 1.0])

    def test_sigmoid_gradient_matches_its_output_formula(self):
        variable = ts.Variable([0.0])
        result = ts.sigmoid(variable)

        ts.backward(result)

        output = result.data.item()
        self.assertAlmostEqual(output, 0.5)
        self.assertAlmostEqual(variable.grad.item(), output * (1.0 - output))

    def test_sqrt_gradient_is_defined_for_positive_values(self):
        variable = ts.Variable([4.0, 9.0])
        result = ts.sqrt(variable)

        ts.backward(ts.sum(result))

        self.assertEqual(result.data.tolist(), [2.0, 3.0])
        self.assertAlmostEqual(variable.grad.tolist()[0], 0.25)
        self.assertAlmostEqual(variable.grad.tolist()[1], 1.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
