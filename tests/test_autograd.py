import unittest

import tensors as ts
from tensors.autograd import Graph, Variable, dot, mean, sum
from tensors.autograd.graph import _reset_graph


class AutogradTests(unittest.TestCase):
    def setUp(self):
        _reset_graph()

    def tearDown(self):
        _reset_graph()

    def test_operation_result_is_owned_by_operation_node(self):
        x = Variable([1.0, 2.0])
        result = x + 2.0

        self.assertEqual(result.node.label, "add")
        self.assertIs(result.node.inputs[0], x.node)

    def test_forward_replays_scalar_and_reduction_operations(self):
        x = Variable([1.0, 2.0, 3.0])
        result = mean(x * 2.0 + 1.0)

        with Graph() as graph:
            replayed = graph.forward(result)

        self.assertEqual(replayed.tolist(), [5.0])

    def test_sum_propagates_upstream_gradient(self):
        x = Variable([1.0, 2.0])
        w = Variable([3.0, 4.0])
        loss = sum(x * w) * 3.0

        with Graph() as graph:
            graph.backward(loss)

        self.assertEqual(x.grad.tolist(), [9.0, 12.0])
        self.assertEqual(w.grad.tolist(), [3.0, 6.0])

    def test_repeated_backward_does_not_reuse_intermediate_gradients(self):
        x = Variable([1.0, 2.0])
        loss = sum(x * x)

        with Graph() as graph:
            graph.backward(loss)
            first = x.grad.tolist()
            graph.backward(loss)
            second = x.grad.tolist()

        self.assertEqual(first, [2.0, 4.0])
        self.assertEqual(second, first)

    def test_forward_refreshes_intermediates_used_by_backward(self):
        x = Variable([2.0])
        square = x * x
        fourth_power = square * square
        x.data = ts.Tensor([3.0])

        with Graph() as graph:
            replayed = graph.forward(fourth_power)
            graph.backward(fourth_power)

        self.assertEqual(replayed.tolist(), [81.0])
        self.assertEqual(x.grad.tolist(), [108.0])

    def test_slice_scatter_backward(self):
        x = Variable([1.0, 2.0, 3.0])
        loss = sum(x[::-1] * 2.0)

        with Graph() as graph:
            graph.backward(loss)

        self.assertEqual(x.grad.tolist(), [2.0, 2.0, 2.0])

    def test_dot_backward_for_2d_tensors(self):
        x = Variable([[1.0, 2.0]])
        w = Variable([[3.0], [4.0]])
        result = dot(x, w)

        with Graph() as graph:
            replayed = graph.forward(result)
            graph.backward(result)

        self.assertEqual(replayed.tolist(), [11.0])
        self.assertEqual(x.grad.tolist(), [3.0, 4.0])
        self.assertEqual(w.grad.tolist(), [1.0, 2.0])

    def test_reverse_division_backward(self):
        x = Variable([2.0, 4.0])
        loss = sum(8.0 / x)

        with Graph() as graph:
            graph.backward(loss)

        self.assertEqual(x.grad.tolist(), [-2.0, -0.5])

    def test_integer_variables_cannot_require_gradients(self):
        with self.assertRaisesRegex(ValueError, "floating-point"):
            Variable(ts.Tensor([1, 2], dtype=ts.int32))

    def test_empty_mean_has_an_empty_gradient(self):
        x = Variable([])
        loss = mean(x)

        with Graph() as graph:
            graph.backward(loss)

        self.assertEqual(x.grad.shape, (0,))
        self.assertEqual(x.grad.tolist(), [])


if __name__ == "__main__":
    unittest.main()
