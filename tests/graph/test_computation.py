import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state


class ComputationTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_computation_owns_backward_pass(self):
        value = ts.Variable([3.0])
        loss = ts.math.sum(value * value)

        ts.graph.Computation(loss).backward()

        self.assertEqual(value.grad.tolist(), [6.0])

    def test_computation_rejects_non_variable_output(self):
        with self.assertRaisesRegex(TypeError, "graph node"):
            ts.graph.Computation(ts.Tensor([1.0]))

    def test_computation_uses_explicit_gradient_seed(self):
        value = ts.Variable([2.0, 3.0])
        result = value * value

        ts.graph.Computation(result).backward(ts.Tensor([10.0, 20.0]))

        self.assertEqual(value.grad.tolist(), [40.0, 120.0])

    def test_computation_casts_gradient_seed_to_output_dtype(self):
        value = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        result = value * value

        ts.graph.Computation(result).backward(
            ts.Tensor([3.0], dtype=ts.float64)
        )

        self.assertIs(value.grad.dtype, ts.float32)
        self.assertEqual(value.grad.tolist(), [12.0])

    def test_computation_restores_input_gradient_dtypes_with_create_graph(self):
        left = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        right = ts.Variable(ts.Tensor([3.0], dtype=ts.float64))
        output = ts.sum(left * right)

        left_gradient, right_gradient = ts.grad(
            output,
            (left, right),
            create_graph=True,
        )
        cross_gradient = ts.grad(ts.sum(left_gradient), right)

        self.assertIs(left_gradient.dtype, ts.float32)
        self.assertIs(right_gradient.dtype, ts.float64)
        self.assertEqual(left_gradient.data.tolist(), [3.0])
        self.assertEqual(right_gradient.data.tolist(), [2.0])
        self.assertEqual(cross_gradient.tolist(), [1.0])

    def test_computation_restores_input_gradient_dtypes_with_create_graph(self):
        left = ts.Variable(ts.Tensor([2.0], dtype=ts.float32))
        right = ts.Variable(ts.Tensor([3.0], dtype=ts.float64))
        output = ts.sum(left * right)

        left_gradient, right_gradient = ts.grad(
            output,
            (left, right),
            create_graph=True,
        )
        cross_gradient = ts.grad(ts.sum(left_gradient), right)

        self.assertIs(left_gradient.dtype, ts.float32)
        self.assertIs(right_gradient.dtype, ts.float64)
        self.assertEqual(left_gradient.data.tolist(), [3.0])
        self.assertEqual(right_gradient.data.tolist(), [2.0])
        self.assertEqual(cross_gradient.tolist(), [1.0])

    def test_computation_rejects_gradient_shape_mismatch(self):
        value = ts.Variable([2.0, 3.0])
        result = value * value

        with self.assertRaisesRegex(ValueError, "Gradient shape"):
            ts.graph.Computation(result).backward(ts.Tensor([1.0]))

    def test_multi_output_graph_exposes_computations_tuple(self):
        @ts.Graph
        def model(x):
            return x + 1.0, x * 2.0

        outputs = model(ts.Tensor([3.0]))

        self.assertEqual(outputs[0].data.tolist(), [4.0])
        self.assertEqual(outputs[1].data.tolist(), [6.0])
        self.assertEqual(len(model.computations), 2)

    def test_single_computation_property_rejects_multi_output_graph(self):
        @ts.Graph
        def model(x):
            return x + 1.0, x * 2.0

        model(ts.Tensor([3.0]))

        with self.assertRaisesRegex(RuntimeError, "multiple outputs"):
            _ = model.computation

    def test_external_loss_backpropagates_into_model_parameters(self):
        class Linear(ts.Graph):
            def __init__(self):
                super().__init__()
                self.weight = ts.Variable([2.0])
                self.bias = ts.Variable([1.0])

            def forward(self, x):
                return x * self.weight + self.bias

        model = Linear()
        loss = ts.math.sum(model(ts.Tensor([3.0])))

        ts.backward(loss)

        self.assertEqual(model.weight.grad.tolist(), [3.0])
        self.assertEqual(model.bias.grad.tolist(), [1.0])

    def test_computation_caches_its_dependency_order(self):
        value = ts.Variable([2.0])
        result = (value + 1.0) * 3.0
        computation = ts.graph.Computation(result)
        cached_order = computation._nodes

        first = computation.nodes
        second = computation.nodes
        first.clear()

        self.assertIs(computation._nodes, cached_order)
        self.assertEqual(second, list(cached_order))
        self.assertEqual(computation.nodes, list(cached_order))
        self.assertEqual(computation.forward().tolist(), [9.0])

    def test_released_computation_rejects_further_work(self):
        value = ts.Variable([2.0])
        result = value * 3.0
        computation = ts.graph.Computation(result)

        computation.release()
        computation.release()

        with self.assertRaisesRegex(RuntimeError, "released"):
            _ = computation.nodes
        with self.assertRaisesRegex(RuntimeError, "released"):
            computation.forward()
        with self.assertRaisesRegex(RuntimeError, "released"):
            computation.backward()


if __name__ == "__main__":
    unittest.main()
