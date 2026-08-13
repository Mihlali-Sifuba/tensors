import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.graph.state import reset_graph_state


class MutationSafetyTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_tensor_version_increments_after_in_place_assignment(self):
        value = ts.Tensor([1.0, 2.0])

        self.assertEqual(value.version, 0)
        value[0] = 3.0

        self.assertEqual(value.version, 1)

    def test_failed_assignment_does_not_change_version(self):
        value = ts.Tensor([1.0])

        with self.assertRaises(IndexError):
            value[3] = 2.0

        self.assertEqual(value.version, 0)

    def test_clone_has_independent_storage_and_version(self):
        original = ts.Tensor([1.0])
        clone = original.clone()
        clone[0] = 2.0

        self.assertEqual(original.tolist(), [1.0])
        self.assertEqual(original.version, 0)
        self.assertEqual(clone.version, 1)

    def test_tensor_metadata_cannot_be_reassigned(self):
        value = ts.Tensor([[1.0]])

        with self.assertRaises(AttributeError):
            value.shape = (1,)
        with self.assertRaises(AttributeError):
            value.ndim = 1
        with self.assertRaises(AttributeError):
            value.dtype = ts.float32

    def test_backward_rejects_in_place_input_mutation(self):
        x = ts.Variable([2.0], name="x")
        output = x * x
        x.data[0] = 3.0

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.backward(output)

    def test_grad_rejects_replaced_input_data_without_changing_grad(self):
        x = ts.Variable([2.0], name="x")
        output = x ** 3.0
        previous = ts.Tensor([17.0])
        x.grad = previous
        x.data = ts.Tensor([3.0])

        with self.assertRaisesRegex(RuntimeError, "Input 0 .*'x'.*modified"):
            ts.grad(output, x)

        self.assertIs(x.grad, previous)

    def test_shared_tensor_mutation_is_detected(self):
        data = ts.Tensor([2.0])
        x = ts.Variable(data, name="x")
        output = x + 1.0
        data[0] = 4.0

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.backward(output)

    def test_mutation_of_unrelated_variable_does_not_invalidate_computation(self):
        x = ts.Variable([2.0])
        unrelated = ts.Variable([7.0])
        output = x * x
        unrelated.data[0] = 8.0

        ts.backward(output)

        self.assertEqual(x.grad.tolist(), [4.0])

    def test_mutated_intermediate_is_detected(self):
        x = ts.Variable([2.0])
        square = x * x
        output = square + 1.0
        square.data[0] = 99.0

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.backward(output)

    def test_mutated_final_output_is_detected(self):
        x = ts.Variable([2.0])
        output = x * x
        output.data[0] = 100.0

        with self.assertRaisesRegex(RuntimeError, "Output of operation 'mul'"):
            ts.backward(output)

    def test_tensor_operands_are_recorded_as_constant_inputs(self):
        x = ts.Variable([2.0])
        coefficient = ts.Tensor([3.0])
        output = ts.sum(x * coefficient)

        self.assertEqual(ts.grad(output, x).tolist(), [3.0])

        coefficient[0] = 4.0
        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.grad(output, x)

    def test_mutable_reduction_axis_is_copied_into_graph_metadata(self):
        x = ts.Variable([[1.0, 2.0], [3.0, 4.0]])
        axes = [0]
        output = ts.sum(x, axis=axes)
        axes[0] = 1

        replayed = Computation(output).forward()

        self.assertEqual(output.node.args["axis"], (0,))
        self.assertEqual(replayed.tolist(), [4.0, 6.0])

    def test_optimizer_step_invalidates_an_old_loss(self):
        weight = ts.Variable([2.0])
        loss = weight * weight
        optimizer = ts.optim.SGD([weight], learning_rate=0.1)
        ts.backward(loss)
        optimizer.step()

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.backward(loss)

        fresh_loss = weight * weight
        ts.backward(fresh_loss)
        self.assertAlmostEqual(weight.grad.item(), 3.2)

    def test_jacobian_and_hessian_reject_stale_inputs(self):
        x = ts.Variable([2.0])
        output = x ** 3.0
        x.data[0] = 3.0

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.jacobian(output, x)
        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.hessian(output, x)

    def test_forward_refreshes_values_and_recorded_input_states(self):
        x = ts.Variable([2.0])
        square = x * x
        output = square + 1.0
        computation = Computation(output)
        x.data[0] = 3.0

        recomputed = computation.forward()
        ts.backward(output)

        self.assertEqual(recomputed.tolist(), [10.0])
        self.assertEqual(x.grad.tolist(), [6.0])

    def test_replacing_data_with_integer_tensor_preserves_grad_invariant(self):
        x = ts.Variable([2.0])

        with self.assertRaisesRegex(ValueError, "floating-point"):
            x.data = ts.Tensor([2], dtype=ts.int32)

        self.assertEqual(x.data.tolist(), [2.0])

    def test_enabling_gradients_on_integer_data_is_rejected(self):
        x = ts.Variable(ts.Tensor([2], dtype=ts.int32), requires_grad=False)

        with self.assertRaisesRegex(ValueError, "floating-point"):
            x.requires_grad = True

        self.assertFalse(x.requires_grad)

    def test_changing_requires_grad_after_forward_is_detected(self):
        x = ts.Variable([2.0])
        output = x * x
        x.requires_grad = False

        with self.assertRaisesRegex(RuntimeError, "modified after its forward pass"):
            ts.backward(output)


if __name__ == "__main__":
    unittest.main()
