import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class ElementwiseMathTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_exp_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 1.0])

        result = ts.exp(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 1.0)
        self.assertAlmostEqual(result.tolist()[1], math.e)

    def test_log_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([1.0, math.e])

        result = ts.log(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.0)
        self.assertAlmostEqual(result.tolist()[1], 1.0)

    def test_sqrt_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([4.0, 9.0])

        result = ts.sqrt(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [2.0, 3.0])

    def test_relu_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([-2.0, 0.0, 3.0])

        result = ts.relu(tensor)

        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.tolist(), [0.0, 0.0, 3.0])

    def test_sigmoid_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 2.0])

        result = ts.sigmoid(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.5)
        self.assertAlmostEqual(result.tolist()[1], 1.0 / (1.0 + math.exp(-2.0)))

    def test_tanh_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 1.0])

        result = ts.tanh(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], 0.0)
        self.assertAlmostEqual(result.tolist()[1], math.tanh(1.0))

    def test_softplus_returns_elementwise_tensor_values(self):
        tensor = ts.Tensor([0.0, 2.0])

        result = ts.softplus(tensor)

        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(result.tolist()[0], math.log(2.0))
        self.assertAlmostEqual(result.tolist()[1], math.log1p(math.exp(2.0)))

    def test_math_namespace_exposes_elementwise_functions(self):
        self.assertAlmostEqual(ts.math.sqrt([4.0]).item(), 2.0)
        self.assertAlmostEqual(ts.math.exp([0.0]).item(), 1.0)
        self.assertAlmostEqual(ts.math.log([math.e]).item(), 1.0)
        self.assertEqual(ts.math.relu([-1.0, 2.0]).tolist(), [0.0, 2.0])
        self.assertAlmostEqual(ts.math.sigmoid([0.0]).item(), 0.5)
        self.assertAlmostEqual(ts.math.tanh([1.0]).item(), math.tanh(1.0))
        self.assertAlmostEqual(ts.math.softplus([0.0]).item(), math.log(2.0))

    def test_integer_inputs_promote_exp_and_log_to_float64(self):
        self.assertIs(ts.sqrt(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.exp(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.log(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.sigmoid(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.tanh(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)
        self.assertIs(ts.softplus(ts.Tensor([1], dtype=ts.int32)).dtype, ts.float64)

    def test_relu_preserves_input_dtype(self):
        self.assertIs(ts.relu(ts.Tensor([-1, 2], dtype=ts.int32)).dtype, ts.int32)
        self.assertIs(ts.relu(ts.Tensor([-1, 2], dtype=ts.float32)).dtype, ts.float32)

    def test_float32_inputs_preserve_dtype(self):
        self.assertIs(ts.sqrt(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.exp(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.log(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.sigmoid(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.tanh(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)
        self.assertIs(ts.softplus(ts.Tensor([1], dtype=ts.float32)).dtype, ts.float32)

    def test_log_rejects_non_positive_values(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            ts.log(ts.Tensor([0.0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            ts.log(ts.Tensor([-1.0]))

    def test_sqrt_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ts.sqrt(ts.Tensor([-1.0]))

    def test_math_namespace_exposes_elementwise_operation_classes(self):
        self.assertAlmostEqual(ts.math.Sqrt.forward(ts.Tensor([4.0])).item(), 2.0)
        self.assertAlmostEqual(ts.math.Exp.forward(ts.Tensor([0.0])).item(), 1.0)
        self.assertAlmostEqual(ts.math.Log.forward(ts.Tensor([math.e])).item(), 1.0)
        self.assertEqual(ts.math.ReLU.forward(ts.Tensor([-1.0, 2.0])).tolist(), [0.0, 2.0])
        self.assertAlmostEqual(ts.math.Sigmoid.forward(ts.Tensor([0.0])).item(), 0.5)
        self.assertAlmostEqual(ts.math.Tanh.forward(ts.Tensor([1.0])).item(), math.tanh(1.0))
        self.assertAlmostEqual(ts.math.Softplus.forward(ts.Tensor([0.0])).item(), math.log(2.0))

    def test_exp_backward(self):
        x = ts.Variable([0.0, 1.0])
        loss = ts.sum(ts.exp(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.tolist()[0], 1.0)
        self.assertAlmostEqual(x.grad.tolist()[1], math.e)

    def test_log_backward(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.sum(ts.log(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.tolist()[0], 1.0)
        self.assertAlmostEqual(x.grad.tolist()[1], 0.5)

    def test_sqrt_backward(self):
        x = ts.Variable([4.0])
        loss = ts.sum(ts.sqrt(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.25)

    def test_relu_backward(self):
        x = ts.Variable([-1.0, 0.0, 2.0])
        loss = ts.sum(ts.relu(x))

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 1.0])

    def test_sigmoid_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.sigmoid(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.25)

    def test_tanh_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.tanh(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 1.0)

    def test_softplus_backward(self):
        x = ts.Variable([0.0])
        loss = ts.sum(ts.softplus(x))

        ts.backward(loss)

        self.assertAlmostEqual(x.grad.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
