import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class VariableTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_variable_wraps_tensor_without_copying_data(self):
        tensor = ts.Tensor([1.0, 2.0])

        variable = ts.Variable(tensor)

        self.assertIs(variable.data, tensor)
        self.assertEqual(variable.shape, (2,))
        self.assertEqual(variable.ndim, 1)
        self.assertEqual(variable.size, 2)
        self.assertIs(variable.dtype, ts.float64)

    def test_variable_can_wrap_another_variable_data(self):
        original = ts.Variable([1.0, 2.0])

        wrapped = ts.Variable(original)

        self.assertIs(wrapped.data, original.data)

    def test_variable_repr_includes_gradient_when_present(self):
        variable = ts.Variable([1.0])
        variable.grad = ts.Tensor([2.0])

        self.assertIn("grad=", repr(variable))

    def test_non_trainable_variable_does_not_accumulate_gradients(self):
        frozen = ts.Variable([2.0], requires_grad=False)
        trainable = ts.Variable([3.0])
        loss = ts.sum(frozen * trainable)

        ts.backward(loss)

        self.assertIsNone(frozen.grad)
        self.assertEqual(trainable.grad.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
