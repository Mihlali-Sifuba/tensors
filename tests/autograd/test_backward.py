import math
import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.ops import Operation
from tensors.graph.state import reset_graph_state


class AutogradTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_operation_result_is_produced_by_an_operation_node(self):
        x = ts.Variable([1.0, 2.0])
        result = x + 2.0

        self.assertEqual(result.node.producer.label, "add")
        self.assertIs(result.node.producer.inputs[0], x.node)

    def test_node_labels_are_derived_and_do_not_control_execution(self):
        x = ts.Variable([2.0])
        result = x * 3.0

        # Labels describe the recorded graph; execution reads the operation.
        self.assertEqual(x.node.label, "var")
        self.assertEqual(result.node.producer.label, "mul")

        replayed = Computation(result).forward()
        ts.backward(result)

        self.assertEqual(replayed.tolist(), [6.0])
        self.assertEqual(x.grad.tolist(), [3.0])

    def test_large_computation_does_not_depend_on_python_recursion(self):
        value = ts.Variable([0.0])
        result = value
        for _ in range(1500):
            result = result + 1.0

        computation = Computation(result)

        # One leaf, then a scalar operand, operation, and result per step.
        self.assertEqual(len(computation.nodes), 1 + 1500 * 3)
        self.assertEqual(computation.forward().tolist(), [1500.0])

    def test_grad_validates_requested_inputs(self):
        value = ts.Variable([2.0])

        with self.assertRaisesRegex(ValueError, "at least one"):
            ts.grad(value * 2.0, ())
        with self.assertRaisesRegex(TypeError, "input 0 must be a Variable"):
            ts.grad(value * 2.0, [ts.Tensor([2.0])])

    def test_forward_replays_scalar_and_reduction_operations(self):
        x = ts.Variable([1.0, 2.0, 3.0])
        result = ts.mean(x * 2.0 + 1.0)

        replayed = Computation(result).forward()

        self.assertEqual(replayed.tolist(), [5.0])

    def test_sum_propagates_upstream_gradient(self):
        x = ts.Variable([1.0, 2.0])
        w = ts.Variable([3.0, 4.0])
        loss = ts.sum(x * w) * 3.0

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [9.0, 12.0])
        self.assertEqual(w.grad.tolist(), [3.0, 6.0])

    def test_broadcast_gradient_accumulation_avoids_temporary_overflow(self):
        value = ts.Variable([1.0])
        output = value + ts.Tensor([0.0, 0.0, 0.0, 0.0])
        seed = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])

        ts.backward(output, seed)

        self.assertEqual(value.grad.tolist(), [0.0])

    def test_shared_branch_accumulation_avoids_temporary_overflow(self):
        value = ts.Variable([1.0])
        output = ts.concat([value, value, value, value])
        seed = ts.Tensor([1.0e308, 1.0e308, -1.0e308, -1.0e308])

        ts.backward(output, seed)

        self.assertEqual(value.grad.tolist(), [0.0])

    def test_repeated_backward_does_not_reuse_intermediate_gradients(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.sum(x * x)

        ts.backward(loss)
        first = x.grad.tolist()
        ts.backward(loss)
        second = x.grad.tolist()

        self.assertEqual(first, [2.0, 4.0])
        self.assertEqual(second, first)

    def test_grad_preserves_a_stale_gradient_for_a_disconnected_input(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        previous = ts.Tensor([7.0])
        y.grad = previous

        self.assertEqual(ts.grad(y * 2.0, y).tolist(), [2.0])
        self.assertIsNone(ts.grad(x * 3.0, y))
        self.assertIs(y.grad, previous)

    def test_grad_does_not_modify_reachable_grad_attributes(self):
        x = ts.Variable([2.0])
        y = ts.Variable([3.0])
        product = x * y
        output = ts.sum(product)
        previous = {
            x: ts.Tensor([10.0]),
            y: ts.Tensor([20.0]),
            product: ts.Tensor([30.0]),
            output: ts.Tensor([40.0]),
        }
        for variable, gradient in previous.items():
            variable.grad = gradient

        x_gradient, y_gradient = ts.grad(output, (x, y))

        self.assertEqual(x_gradient.tolist(), [3.0])
        self.assertEqual(y_gradient.tolist(), [2.0])
        for variable, gradient in previous.items():
            self.assertIs(variable.grad, gradient)

    def test_create_graph_grad_does_not_modify_grad_attributes(self):
        x = ts.Variable([2.0])
        output = x ** 3.0
        previous_x_gradient = ts.Tensor([7.0])
        previous_output_gradient = ts.Tensor([8.0])
        x.grad = previous_x_gradient
        output.grad = previous_output_gradient

        result = ts.grad(output, x, create_graph=True)

        self.assertEqual(result.data.tolist(), [12.0])
        self.assertIs(x.grad, previous_x_gradient)
        self.assertIs(output.grad, previous_output_gradient)

    def test_backward_rejects_an_invalid_gradient_count(self):
        class BrokenOperation(Operation):
            name = "broken"

            def forward(self, value):
                return value

            def backward(self, gradient, value, *, needs_input_grad):
                return []

        value = ts.Variable([2.0])
        operation = BrokenOperation()
        output = value._apply_operation(operation, (value,))

        with self.assertRaisesRegex(RuntimeError, "returned 0 gradients for 1 inputs"):
            ts.backward(output)

    def test_failed_backward_does_not_partially_replace_gradients(self):
        class BrokenOperation(Operation):
            name = "broken"

            def forward(self, value):
                return value

            def backward(self, gradient, value, *, needs_input_grad):
                raise RuntimeError("deliberate failure")

        value = ts.Variable([2.0])
        operation = BrokenOperation()
        output = value._apply_operation(operation, (value,))
        previous_value_gradient = ts.Tensor([7.0])
        previous_output_gradient = ts.Tensor([8.0])
        value.grad = previous_value_gradient
        output.grad = previous_output_gradient

        with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
            ts.backward(output)

        self.assertIs(value.grad, previous_value_gradient)
        self.assertIs(output.grad, previous_output_gradient)

    def test_forward_refreshes_intermediates_used_by_backward(self):
        x = ts.Variable([2.0])
        square = x * x
        fourth_power = square * square
        x.data = ts.Tensor([3.0])

        replayed = Computation(fourth_power).forward()
        ts.backward(fourth_power)

        self.assertEqual(replayed.tolist(), [81.0])
        self.assertEqual(x.grad.tolist(), [108.0])

    def test_slice_scatter_backward(self):
        x = ts.Variable([1.0, 2.0, 3.0])
        loss = ts.sum(x[::-1] * 2.0)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [2.0, 2.0, 2.0])

    def test_integer_indexing_returns_scalar_variable(self):
        x = ts.Variable([[1.0, 2.0], [3.0, 4.0]])

        selected = x[1, 0]
        ts.backward(selected)

        self.assertEqual(selected.shape, ())
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_dot_backward_for_2d_tensors(self):
        x = ts.Variable([[1.0, 2.0]])
        w = ts.Variable([[3.0], [4.0]])
        result = ts.linalg.dot(x, w)

        replayed = Computation(result).forward()
        ts.backward(result)

        self.assertEqual(replayed.tolist(), [11.0])
        self.assertEqual(x.grad.tolist(), [3.0, 4.0])
        self.assertEqual(w.grad.tolist(), [1.0, 2.0])

    def test_reverse_division_backward(self):
        x = ts.Variable([2.0, 4.0])
        loss = ts.sum(8.0 / x)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [-2.0, -0.5])

    def test_integer_variables_cannot_require_gradients(self):
        with self.assertRaisesRegex(ValueError, "floating-point"):
            ts.Variable(ts.Tensor([1, 2], dtype=ts.int32))

    def test_empty_mean_has_an_empty_gradient(self):
        x = ts.Variable([])
        loss = ts.mean(x)

        ts.backward(loss)

        self.assertEqual(x.grad.shape, (0,))
        self.assertEqual(x.grad.tolist(), [])

    def test_math_namespace_keeps_variable_reductions_differentiable(self):
        x = ts.Variable([1.0, 2.0])
        loss = ts.math.sum(x * 3.0)

        ts.backward(loss)

        self.assertEqual(x.grad.tolist(), [3.0, 3.0])

    def test_sgd_updates_external_model_parameters(self):
        weight = ts.Variable([1.0])
        loss = ts.math.sum(weight * 2.0)
        optimizer = ts.optim.SGD([weight], learning_rate=0.1)

        ts.backward(loss)
        optimizer.step()

        self.assertEqual(weight.data.tolist(), [0.8])
        optimizer.zero_grad()
        self.assertIsNone(weight.grad)


class AdditionVjpTests(unittest.TestCase):
    """Addition's single ``backward``, through the public API.

    ``Add`` defines its derivative once, against operations rather than
    against Tensors, so an ordinary reverse pass and one building a
    derivative graph run the same statements over different operands. Every
    case below is therefore asserted in both modes, and the recorded mode is
    also replayed, because a recorded VJP is a program and not an answer.

    Addition is the only operation migrated so far. Expressions here are
    built from ``+`` alone so that what fails is addition's own behaviour.
    """

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _gradients(self, left, right, seed, *, create_graph, dtype=ts.float64):
        """Both VJPs of ``left + right`` at ``seed``, in one reverse mode."""
        a = ts.Variable(ts.Tensor(left, dtype=dtype), name="a")
        b = ts.Variable(ts.Tensor(right, dtype=dtype), name="b")
        first, second = ts.grad(
            a + b,
            [a, b],
            grad_outputs=ts.Tensor(seed, dtype=dtype),
            create_graph=create_graph,
        )
        if create_graph:
            first, second = first.data, second.data
        return first, second

    def _assertBothModes(self, left, right, seed, expected, dtype=ts.float64):
        """Assert the values, shapes and dtypes of both VJPs, in both modes."""
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                reset_graph_state()
                produced = self._gradients(
                    left, right, seed, create_graph=create_graph, dtype=dtype
                )
                for index, (gradient, operand) in enumerate(
                    zip(produced, (left, right))
                ):
                    with self.subTest(input=index):
                        wanted = ts.Tensor(operand, dtype=dtype)
                        self.assertEqual(gradient.tolist(), expected[index])
                        self.assertEqual(gradient.shape, wanted.shape)
                        self.assertEqual(gradient.dtype, dtype)

    def test_equal_shapes_hand_the_gradient_to_both_operands(self):
        self._assertBothModes(
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            expected=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        )

    def test_a_stretched_axis_sums_the_positions_it_fed(self):
        """(1, 3) + (2, 1): each operand reduces along a different axis."""
        self._assertBothModes(
            [[1.0, 2.0, 3.0]],
            [[10.0], [20.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[5.0, 7.0, 9.0], [6.0, 15.0]],
        )

    def test_an_added_leading_axis_is_summed_away(self):
        """(2, 3) + (3,): the rank the broadcast added is reduced out."""
        self._assertBothModes(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [1.0, 2.0, 3.0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [5.0, 7.0, 9.0]],
        )

    def test_a_fully_reduced_operand_keeps_its_own_shape(self):
        self._assertBothModes(
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            expected=[[1.0, 2.0, 3.0, 4.0], [10.0]],
        )

    def test_float32_gradients_keep_their_dtype(self):
        self._assertBothModes(
            [[1.0, 2.0, 3.0]],
            [[10.0], [20.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[5.0, 7.0, 9.0], [6.0, 15.0]],
            dtype=ts.float32,
        )

    def test_only_one_operand_may_be_requested(self):
        """An unrequested operand is skipped, not calculated and discarded."""
        for create_graph in (False, True):
            for wanted_index in (0, 1):
                with self.subTest(create_graph=create_graph, input=wanted_index):
                    reset_graph_state()
                    operands = [
                        ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]])),
                        ts.Variable(ts.Tensor([[10.0], [20.0]])),
                    ]
                    operands[1 - wanted_index].requires_grad = False
                    seed = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    result = ts.grad(
                        operands[0] + operands[1],
                        operands[wanted_index],
                        grad_outputs=seed,
                        create_graph=create_graph,
                    )
                    produced = result.data if create_graph else result
                    expected = [[5.0, 7.0, 9.0], [6.0, 15.0]][wanted_index]
                    self.assertEqual(produced.tolist(), expected)
                    self.assertEqual(
                        produced.shape, operands[wanted_index].data.shape
                    )
                    self.assertIsNone(
                        ts.grad(
                            operands[0] + operands[1],
                            operands[1 - wanted_index],
                            grad_outputs=seed,
                        )
                    )

    def test_a_repeated_operand_accumulates_both_contributions(self):
        """``x + x`` reaches the same Variable through both input slots."""
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                reset_graph_state()
                x = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]]))
                result = ts.grad(
                    x + x,
                    x,
                    grad_outputs=ts.Tensor([[1.0, 2.0, 3.0]]),
                    create_graph=create_graph,
                )
                produced = result.data if create_graph else result
                self.assertEqual(produced.tolist(), [2.0, 4.0, 6.0])
                self.assertEqual(tuple(produced.shape), (1, 3))

    def test_a_repeated_operand_accumulates_across_a_broadcast(self):
        """``(x + y) + x`` reduces one contribution and passes the other."""
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                reset_graph_state()
                x = ts.Variable(ts.Tensor([1.0, 2.0, 3.0]))
                y = ts.Variable(ts.Tensor([[1.0], [2.0]]))
                result = ts.grad(
                    (x + y) + x,
                    x,
                    grad_outputs=ts.Tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
                    create_graph=create_graph,
                )
                produced = result.data if create_graph else result
                # Both contributions reduce (2, 3) to (3,): 2 + 2 per position.
                self.assertEqual(produced.tolist(), [4.0, 4.0, 4.0])
                self.assertEqual(tuple(produced.shape), (3,))

    def test_a_scalar_operand_leaves_the_gradient_untouched(self):
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                reset_graph_state()
                x = ts.Variable(ts.Tensor([1.0, 2.0]))
                result = ts.grad(
                    x + 3.0,
                    x,
                    grad_outputs=ts.Tensor([1.0, 2.0]),
                    create_graph=create_graph,
                )
                produced = result.data if create_graph else result
                self.assertEqual(produced.tolist(), [1.0, 2.0])

    def test_the_recorded_gradient_replays_with_a_changed_seed(self):
        """The recorded VJP is a program: re-run it, do not re-derive it."""
        a = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]]))
        b = ts.Variable(ts.Tensor([[10.0], [20.0]]))
        seed = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        first, second = ts.grad(a + b, [a, b], grad_outputs=seed, create_graph=True)

        left_program = Computation(first)
        right_program = Computation(second)
        self.assertEqual(left_program.forward().tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(right_program.forward().tolist(), [6.0, 15.0])

        seed.data = ts.Tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        replayed_left = left_program.forward()
        replayed_right = right_program.forward()

        self.assertEqual(replayed_left.tolist(), [50.0, 70.0, 90.0])
        self.assertEqual(tuple(replayed_left.shape), (1, 3))
        self.assertEqual(replayed_right.tolist(), [60.0, 150.0])
        self.assertEqual(tuple(replayed_right.shape), (2, 1))

    def test_a_seed_derivative_without_a_broadcast_is_the_identity(self):
        a = ts.Variable(ts.Tensor([1.0, 2.0]))
        b = ts.Variable(ts.Tensor([3.0, 4.0]))
        seed = ts.Variable(ts.Tensor([1.0, 2.0]))
        first, _ = ts.grad(a + b, [a, b], grad_outputs=seed, create_graph=True)

        by_seed = ts.grad(first, seed, grad_outputs=ts.Tensor([5.0, 7.0]))
        self.assertEqual(by_seed.tolist(), [5.0, 7.0])
        self.assertEqual(tuple(by_seed.shape), (2,))


class AdditionSeedDerivativeTests(unittest.TestCase):
    """Differentiating addition's own broadcast gradient, on every backend.

    Expected values are the derivative rules, not another backend's output.
    For ``z = a + b`` the derivative with respect to either operand is one, so
    a stretched operand's VJP is the sum of the output positions it fed. That
    sum is linear, so differentiating it by the upstream seed spreads the
    cotangent back over exactly the positions each sum consumed.

    Two broadcasts are covered because they reach different dependencies. A
    stretched singleton axis reduces to a shape the reduction already has, so
    the VJP records the reduction alone. A removed leading axis reduces to a
    lower rank, so the VJP records the reduction and then a relabelling, and
    the recorded derivative has to pass through both.

    Every stage is checked for value, shape, dtype and residency: the reverse
    pass runs where the selection says, and a backend that is not installed is
    reported rather than passed over.
    """

    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        """Skip this subtest, visibly, when the backend is not installed."""
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def assertProduced(self, produced, *, values, shape, backend, dtype=ts.float64):
        self.assertEqual(produced.tolist(), values)
        self.assertEqual(tuple(produced.shape), shape)
        self.assertEqual(produced.dtype, dtype)
        self.assertEqual(produced.backend_storage.kind, backend)

    def _stretched_axis_gradient(self, dtype=ts.float64):
        """``(1, 3) + (2, 1)``: each operand is stretched along one axis."""
        a = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]], dtype=dtype), name="a")
        b = ts.Variable(ts.Tensor([[10.0], [20.0]], dtype=dtype), name="b")
        seed = ts.Variable(
            ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype), name="seed"
        )
        left, right = ts.grad(
            a + b, [a, b], grad_outputs=seed, create_graph=True
        )
        return left, right, seed

    def _leading_axis_gradient(self, dtype=ts.float64):
        """``(2, 3) + (3,)``: the vector's VJP loses the axis it gained."""
        m = ts.Variable(
            ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype), name="m"
        )
        v = ts.Variable(ts.Tensor([1.0, 2.0, 3.0], dtype=dtype), name="v")
        seed = ts.Variable(
            ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype), name="seed"
        )
        _, vector = ts.grad(m + v, [m, v], grad_outputs=seed, create_graph=True)
        return vector, seed

    def test_a_stretched_axis_gradient_differentiates_by_its_seed(self):
        """Sum down axis 0, then its derivative spreads back up axis 0."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        left, right, seed = self._stretched_axis_gradient()
                        self.assertProduced(
                            left.data,
                            values=[5.0, 7.0, 9.0],
                            shape=(1, 3),
                            backend=backend,
                        )
                        self.assertProduced(
                            right.data,
                            values=[6.0, 15.0],
                            shape=(2, 1),
                            backend=backend,
                        )

                        # Each seed position feeds exactly one summed
                        # position, so the cotangent arrives unscaled and is
                        # copied to every row the sum consumed.
                        by_seed = ts.grad(
                            left,
                            seed,
                            grad_outputs=ts.Tensor([[2.0, 3.0, 5.0]]),
                            create_graph=create_graph,
                        )
                        produced = by_seed.data if create_graph else by_seed
                        self.assertProduced(
                            produced,
                            values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                            shape=(2, 3),
                            backend=backend,
                        )

                        # The other operand summed the other axis, so its
                        # cotangent spreads along the columns instead.
                        by_seed_right = ts.grad(
                            right,
                            seed,
                            grad_outputs=ts.Tensor([[2.0], [3.0]]),
                            create_graph=create_graph,
                        )
                        produced = (
                            by_seed_right.data if create_graph else by_seed_right
                        )
                        self.assertProduced(
                            produced,
                            values=[2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
                            shape=(2, 3),
                            backend=backend,
                        )

    def test_a_removed_leading_axis_gradient_differentiates_by_its_seed(self):
        """The VJP reduced and then relabelled, so both are differentiated."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        vector, seed = self._leading_axis_gradient()
                        self.assertProduced(
                            vector.data,
                            values=[5.0, 7.0, 9.0],
                            shape=(3,),
                            backend=backend,
                        )

                        by_seed = ts.grad(
                            vector,
                            seed,
                            grad_outputs=ts.Tensor([2.0, 3.0, 5.0]),
                            create_graph=create_graph,
                        )
                        produced = by_seed.data if create_graph else by_seed
                        self.assertProduced(
                            produced,
                            values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                            shape=(2, 3),
                            backend=backend,
                        )

    def test_float32_seed_derivatives_keep_their_dtype(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    vector, seed = self._leading_axis_gradient(dtype=ts.float32)
                    by_seed = ts.grad(
                        vector,
                        seed,
                        grad_outputs=ts.Tensor([2.0, 3.0, 5.0], dtype=ts.float32),
                        create_graph=True,
                    )
                    self.assertProduced(
                        by_seed.data,
                        values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                        shape=(2, 3),
                        backend=backend,
                        dtype=ts.float32,
                    )

    def test_the_recorded_seed_derivative_stays_connected_to_its_cotangent(self):
        """A differentiable second seed must reach the new derivative.

        The second derivative spreads the second seed over the two rows the
        first sum consumed, so differentiating it back by that seed with a
        cotangent of ones counts those rows: two per position.
        """
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    left, _, seed = self._stretched_axis_gradient()
                    second_seed = ts.Variable(
                        ts.Tensor([[2.0, 3.0, 5.0]]), name="second_seed"
                    )
                    built = ts.grad(
                        left, seed, grad_outputs=second_seed, create_graph=True
                    )
                    self.assertProduced(
                        built.data,
                        values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                        shape=(2, 3),
                        backend=backend,
                    )

                    back = ts.grad(
                        built,
                        second_seed,
                        grad_outputs=ts.Tensor(
                            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                        ),
                    )
                    self.assertProduced(
                        back,
                        values=[2.0, 2.0, 2.0],
                        shape=(1, 3),
                        backend=backend,
                    )

    def test_the_recorded_seed_derivative_replays_with_a_changed_cotangent(self):
        """Re-run the recorded second derivative; do not rebuild it."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    left, _, seed = self._stretched_axis_gradient()
                    second_seed = ts.Variable(ts.Tensor([[2.0, 3.0, 5.0]]))
                    built = ts.grad(
                        left, seed, grad_outputs=second_seed, create_graph=True
                    )
                    program = Computation(built)
                    self.assertProduced(
                        program.forward(),
                        values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                        shape=(2, 3),
                        backend=backend,
                    )

                    second_seed.data = ts.Tensor([[7.0, 11.0, 13.0]])
                    replayed = program.forward()
                self.assertProduced(
                    replayed,
                    values=[7.0, 11.0, 13.0, 7.0, 11.0, 13.0],
                    shape=(2, 3),
                    backend=backend,
                )

    def test_the_add_chain_differentiates_a_third_time(self):
        """The level that used to stop inside multiplication.

        Add's VJP records a reduction; the reduction's VJP records a
        multiplication by ones; so a third recorded derivative lands in
        ``Mul.backward``. Each seed position feeds two rows of the first
        sum, so differentiating the spread back by its own cotangent of ones
        counts those rows: two everywhere, whichever broadcast was reduced.
        """
        for backend in self.BACKENDS:
            for label in ("stretched", "leading"):
                with self.subTest(backend=backend, broadcast=label):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        if label == "stretched":
                            first, _, seed = self._stretched_axis_gradient()
                            cotangent = ts.Tensor([[2.0, 3.0, 5.0]])
                            shape = (1, 3)
                        else:
                            first, seed = self._leading_axis_gradient()
                            cotangent = ts.Tensor([2.0, 3.0, 5.0])
                            shape = (3,)
                        second_seed = ts.Variable(cotangent)
                        built = ts.grad(
                            first, seed, grad_outputs=second_seed, create_graph=True
                        )
                        self.assertProduced(
                            built.data,
                            values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                            shape=(2, 3),
                            backend=backend,
                        )

                        third = ts.grad(
                            built,
                            second_seed,
                            grad_outputs=ts.Tensor(
                                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                            ),
                            create_graph=True,
                        )
                    self.assertProduced(
                        third.data,
                        values=[2.0, 2.0, 2.0],
                        shape=shape,
                        backend=backend,
                    )

    def test_a_removed_leading_axis_derivative_replays_with_a_changed_cotangent(
        self,
    ):
        """The same, through the relabelling the lower-rank operand records."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    vector, seed = self._leading_axis_gradient()
                    second_seed = ts.Variable(ts.Tensor([2.0, 3.0, 5.0]))
                    built = ts.grad(
                        vector, seed, grad_outputs=second_seed, create_graph=True
                    )
                    program = Computation(built)
                    self.assertProduced(
                        program.forward(),
                        values=[2.0, 3.0, 5.0, 2.0, 3.0, 5.0],
                        shape=(2, 3),
                        backend=backend,
                    )

                    second_seed.data = ts.Tensor([7.0, 11.0, 13.0])
                    replayed = program.forward()
                self.assertProduced(
                    replayed,
                    values=[7.0, 11.0, 13.0, 7.0, 11.0, 13.0],
                    shape=(2, 3),
                    backend=backend,
                )


class AdditionVjpBoundaryTests(unittest.TestCase):
    """What addition must not have grown back."""

    def test_addition_and_its_dependencies_define_exactly_one_derivative(self):
        from tensors.operations.vjp import ProductSumToShape
        from tensors.operations.manipulation.reshape import Reshape
        from tensors.operations.reductions.sum import Sum
        from tensors.ops import Add, Mul, Neg, Pow, Sub

        for operation in (
            Add, Sum, Reshape, Mul, ProductSumToShape, Sub, Neg, Pow
        ):
            with self.subTest(operation=operation.name):
                self.assertIs(operation.backward_graph, Operation.backward_graph)
                self.assertNotIn("backward_graph", vars(operation))

    def test_the_sum_reduction_carries_no_separate_implementation(self):
        """``Sum.forward`` holds the reduction; no helper stands beside it."""
        import tensors.operations.reductions.sum as module

        self.assertFalse(hasattr(module, "_sum_impl"))
        self.assertEqual(
            ts.sum(ts.Tensor([1.0, 2.0, 3.0])).tolist(), [6.0], "full reduction"
        )
        self.assertEqual(
            tuple(ts.sum(ts.Tensor([1.0, 2.0, 3.0])).shape),
            (1,),
            "a full reduction without keepdims stays a one-element vector",
        )

    def test_no_compatibility_marker_selects_the_reverse_method(self):
        """Both reverse modes call ``backward``; nothing chooses between them."""
        import inspect

        from tensors.graph.computation import computation as module

        source = inspect.getsource(module.Computation._backward_graph)
        self.assertIn("operation.backward(", source)
        # ``_backward_graph`` is this method's own name; the operation call
        # inside it is what must no longer reach a second derivative method.
        self.assertNotIn(".backward_graph(", source)
        self.assertNotIn("backward_accepts_graph_operands", source)
        self.assertFalse(hasattr(Operation, "backward_accepts_graph_operands"))


class MultiplicationVjpTests(unittest.TestCase):
    """Multiplication's single ``backward``, on every backend.

    Expected values are the product rule: d(a*b)/da is b, so each operand's
    VJP is the upstream gradient times the other operand, summed back over
    the axes the forward broadcast stretched. Nothing here reads one
    backend's answer to judge another's.

    The two steps stay one operation. Forming the products first can
    overflow to infinities that cancel to NaN, or underflow to zero, where
    the exact reduced sum is representable, so the fused reduction is kept
    and the cases that exercise it are below.
    """

    BACKENDS = ("python", "numpy", "cuda")

    #: Lengths spanning the workload sizes the reverse pass used to branch
    #: on. A repeated operand's contributions are stacked and reduced before
    #: they are used, and both steps once answered in Python for small work,
    #: so the same mathematical program behaved differently at different
    #: sizes. These cover either side of that boundary and below it.
    SIZES = (1, 2, 3, 31, 32, 64)

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def assertProduced(self, produced, *, values, shape, backend, dtype=ts.float64):
        self.assertEqual(produced.tolist(), values)
        self.assertEqual(tuple(produced.shape), shape)
        self.assertEqual(produced.dtype, dtype)
        self.assertEqual(produced.backend_storage.kind, backend)

    def _assertBothVjps(self, left, right, seed, expected, shapes, dtype=ts.float64):
        """Both VJPs, in both reverse modes, on every available backend."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        a = ts.Variable(ts.Tensor(left, dtype=dtype), name="a")
                        b = ts.Variable(ts.Tensor(right, dtype=dtype), name="b")
                        produced = ts.grad(
                            a * b,
                            [a, b],
                            grad_outputs=ts.Tensor(seed, dtype=dtype),
                            create_graph=create_graph,
                        )
                        if create_graph:
                            produced = [item.data for item in produced]
                    for index, item in enumerate(produced):
                        with self.subTest(input=index):
                            self.assertProduced(
                                item,
                                values=expected[index],
                                shape=shapes[index],
                                backend=backend,
                                dtype=dtype,
                            )

    def test_equal_shapes_weight_each_operand_by_the_other(self):
        self._assertBothVjps(
            [[1.5, -2.5], [3.0, 4.0]],
            [[0.5, 2.0], [-1.0, 8.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            expected=[[0.5, 4.0, -3.0, 32.0], [1.5, -5.0, 9.0, 16.0]],
            shapes=[(2, 2), (2, 2)],
        )

    def test_a_stretched_singleton_axis_sums_the_products_it_fed(self):
        """``(1, 3) * (2, 1)``: each operand reduces the other's axis."""
        self._assertBothVjps(
            [[1.5, -2.5, 3.0]],
            [[2.0], [-4.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[-14.0, -16.0, -18.0], [5.5, 11.5]],
            shapes=[(1, 3), (2, 1)],
        )

    def test_an_added_leading_axis_is_summed_away(self):
        """``(2, 3) * (3,)``: the rank the broadcast added is reduced out."""
        self._assertBothVjps(
            [[1.5, -2.5, 3.0], [2.0, 4.0, 6.0]],
            [0.5, 2.0, -1.0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[0.5, 4.0, -3.0, 2.0, 10.0, -6.0], [9.5, 15.0, 45.0]],
            shapes=[(2, 3), (3,)],
        )

    def test_float32_products_keep_their_dtype(self):
        self._assertBothVjps(
            [[1.5, -2.5, 3.0]],
            [[2.0], [-4.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[-14.0, -16.0, -18.0], [5.5, 11.5]],
            shapes=[(1, 3), (2, 1)],
            dtype=ts.float32,
        )

    def test_a_scalar_factor_scales_the_gradient(self):
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        x = ts.Variable(ts.Tensor([1.5, -2.5]))
                        result = ts.grad(
                            x * 3.0,
                            x,
                            grad_outputs=ts.Tensor([1.0, 2.0]),
                            create_graph=create_graph,
                        )
                        produced = result.data if create_graph else result
                    self.assertProduced(
                        produced, values=[3.0, 6.0], shape=(2,), backend=backend
                    )

    def test_only_one_operand_may_be_requested(self):
        """An unrequested operand's product is never formed."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                for wanted in (0, 1):
                    with self.subTest(
                        backend=backend, create_graph=create_graph, input=wanted
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            operands = [
                                ts.Variable(ts.Tensor([[1.5, -2.5, 3.0]])),
                                ts.Variable(ts.Tensor([[2.0], [-4.0]])),
                            ]
                            operands[1 - wanted].requires_grad = False
                            result = ts.grad(
                                operands[0] * operands[1],
                                operands[wanted],
                                grad_outputs=ts.Tensor(
                                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                                ),
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        expected = [[-14.0, -16.0, -18.0], [5.5, 11.5]][wanted]
                        shape = [(1, 3), (2, 1)][wanted]
                        self.assertProduced(
                            produced, values=expected, shape=shape, backend=backend
                        )

    def test_a_repeated_operand_accumulates_both_products(self):
        """``x * x`` reaches one Variable twice, so its VJP is ``2x``.

        The two contributions are stacked and reduced before they are used,
        at whatever size the expression happens to have, so every length is
        the same program and must answer the same way in the same place.
        """
        for backend in self.BACKENDS:
            for size in self.SIZES:
                for create_graph in (False, True):
                    with self.subTest(
                        backend=backend, size=size, create_graph=create_graph
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            x = ts.Variable(ts.full((size,), 3.0, dtype=ts.float64))
                            result = ts.grad(
                                x * x,
                                x,
                                grad_outputs=ts.full((size,), 1.0, dtype=ts.float64),
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        self.assertProduced(
                            produced,
                            values=[6.0] * size,
                            shape=(size,),
                            backend=backend,
                        )

    def test_a_repeated_operand_in_a_cube_accumulates_three_products(self):
        """``x * x * x`` gives ``3x**2``, at every length."""
        for backend in self.BACKENDS:
            for size in self.SIZES:
                for create_graph in (False, True):
                    with self.subTest(
                        backend=backend, size=size, create_graph=create_graph
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            x = ts.Variable(ts.full((size,), 2.0, dtype=ts.float64))
                            result = ts.grad(
                                x * x * x,
                                x,
                                grad_outputs=ts.full((size,), 1.0, dtype=ts.float64),
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        self.assertProduced(
                            produced,
                            values=[12.0] * size,
                            shape=(size,),
                            backend=backend,
                        )

    def test_a_nonconstant_higher_derivative_stays_connected_to_its_input(self):
        """``x**3`` through three recorded derivatives: 3x^2, 6x, then 6.

        Each level is rebuilt from the recorded one before it, so a level
        that had lost its connection to ``x`` would answer zero rather than a
        value that still moves with ``x``.
        """
        for backend in self.BACKENDS:
            for size in self.SIZES:
                with self.subTest(backend=backend, size=size):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        x = ts.Variable(ts.full((size,), 2.0, dtype=ts.float64))
                        ones = ts.full((size,), 1.0, dtype=ts.float64)
                        first = ts.grad(
                            x * x * x, x, grad_outputs=ones, create_graph=True
                        )
                        second = ts.grad(
                            first, x, grad_outputs=ones, create_graph=True
                        )
                        third = ts.grad(
                            second, x, grad_outputs=ones, create_graph=True
                        )
                    for level, expected in (
                        (first, 12.0),  # 3x^2 at x = 2
                        (second, 12.0),  # 6x   at x = 2
                        (third, 6.0),  # 6
                    ):
                        self.assertProduced(
                            level.data,
                            values=[expected] * size,
                            shape=(size,),
                            backend=backend,
                        )

    def test_a_recorded_derivative_replays_with_a_changed_input(self):
        """Re-run the recorded 3x^2; do not rebuild it."""
        for backend in self.BACKENDS:
            for size in self.SIZES:
                with self.subTest(backend=backend, size=size):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        x = ts.Variable(ts.full((size,), 2.0, dtype=ts.float64))
                        ones = ts.full((size,), 1.0, dtype=ts.float64)
                        first = ts.grad(
                            x * x * x, x, grad_outputs=ones, create_graph=True
                        )
                        program = Computation(first)
                        self.assertProduced(
                            program.forward(),
                            values=[12.0] * size,
                            shape=(size,),
                            backend=backend,
                        )

                        x.data = ts.full((size,), 5.0, dtype=ts.float64)
                        replayed = program.forward()
                    self.assertProduced(
                        replayed,
                        values=[75.0] * size,  # 3x^2 at x = 5
                        shape=(size,),
                        backend=backend,
                    )

    def test_a_recorded_vjp_differentiates_by_its_upstream_seed(self):
        """d(seed * b)/d(seed) is b, recovered through the fused reduction."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([[1.5, -2.5, 3.0]]))
                    b = ts.Variable(ts.Tensor([[2.0], [-4.0]]), requires_grad=False)
                    seed = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
                    first = ts.grad(
                        a * b, a, grad_outputs=seed, create_graph=True
                    )
                    self.assertProduced(
                        first.data,
                        values=[-14.0, -16.0, -18.0],
                        shape=(1, 3),
                        backend=backend,
                    )

                    # The VJP is linear in the seed, so its derivative is the
                    # other operand broadcast over the summed axis.
                    by_seed = ts.grad(
                        first, seed, grad_outputs=ts.Tensor([[1.0, 1.0, 1.0]])
                    )
                self.assertProduced(
                    by_seed,
                    values=[2.0, 2.0, 2.0, -4.0, -4.0, -4.0],
                    shape=(2, 3),
                    backend=backend,
                )

    def test_the_fused_reduction_preserves_exact_cancellation(self):
        """Products that overflow, reduced to a result that does not.

        ``2 * 1e308`` is not representable, so a multiply followed by a
        separate reduction gives infinities that cancel to NaN. The fused
        reduction groups the factors before rounding and answers exactly.
        """
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    value = ts.Variable(ts.Tensor([0.0]))
                    factor = ts.Variable(
                        ts.Tensor([1.0e308, -1.0e308]), requires_grad=False
                    )
                    seed = ts.Variable(ts.Tensor([2.0, 2.0]))
                    derivative = ts.grad(
                        value * factor, value, grad_outputs=seed, create_graph=True
                    )
                    self.assertProduced(
                        derivative.data, values=[0.0], shape=(1,), backend=backend
                    )

                    by_seed = ts.grad(derivative, seed)
                self.assertProduced(
                    by_seed,
                    values=[1.0e308, -1.0e308],
                    shape=(2,),
                    backend=backend,
                )

    def test_a_nonfinite_factor_is_computed_or_explicitly_refused(self):
        """An infinity in the fused reduction is a specified refusal.

        The array kernels cannot carry an infinity through their scaled
        accumulation, so they decline, and a decline under an explicit
        selection is reported rather than answered somewhere else. The
        Python backend accumulates exactly and returns the value. Both are
        the specified behaviour; neither is a fallback for the other.

        The other operand's VJP has no infinite factor, so it is unaffected
        on every backend, which is what makes this a property of the
        reduction and not of the expression.
        """
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([1.0, 2.0]))
                    b = ts.Variable(ts.Tensor([math.inf, 1.0]))
                    seed = ts.Tensor([1.0, 1.0])

                    if backend == "python":
                        produced = ts.grad(a * b, a, grad_outputs=seed)
                        self.assertEqual(produced.tolist(), [math.inf, 1.0])
                        self.assertEqual(produced.backend_storage.kind, backend)
                    else:
                        with self.assertRaises(
                            ts.BackendOperationUnsupportedError
                        ) as raised:
                            ts.grad(a * b, a, grad_outputs=seed)
                        message = str(raised.exception)
                        self.assertIn(backend, message)
                        self.assertIn("sum_products_to_shape", message)

                    # d/db is seed * a, which is finite everywhere.
                    other = ts.grad(a * b, b, grad_outputs=seed)
                self.assertProduced(
                    other, values=[1.0, 2.0], shape=(2,), backend=backend
                )


class SubtractionVjpTests(unittest.TestCase):
    """Subtraction's single ``backward``, on every backend.

    Expected values are the derivative rules: d(a - b)/da is one and
    d(a - b)/db is minus one, so each VJP is the upstream gradient — negated
    for the right operand — summed back over the axes the forward broadcast
    stretched. Nothing here reads one backend's answer to judge another's.
    """

    BACKENDS = ("python", "numpy", "cuda")
    SIZES = (1, 2, 3, 31, 32, 64)

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def assertProduced(self, produced, *, values, shape, backend, dtype=ts.float64):
        self.assertEqual(produced.tolist(), values)
        self.assertEqual(tuple(produced.shape), shape)
        self.assertEqual(produced.dtype, dtype)
        self.assertEqual(produced.backend_storage.kind, backend)

    def _assertBothVjps(self, left, right, seed, expected, shapes, dtype=ts.float64):
        """Both VJPs, in both reverse modes, on every available backend."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        a = ts.Variable(ts.Tensor(left, dtype=dtype), name="a")
                        b = ts.Variable(ts.Tensor(right, dtype=dtype), name="b")
                        produced = ts.grad(
                            a - b,
                            [a, b],
                            grad_outputs=ts.Tensor(seed, dtype=dtype),
                            create_graph=create_graph,
                        )
                        if create_graph:
                            produced = [item.data for item in produced]
                    for index, item in enumerate(produced):
                        with self.subTest(input=index):
                            self.assertProduced(
                                item,
                                values=expected[index],
                                shape=shapes[index],
                                backend=backend,
                                dtype=dtype,
                            )

    def test_equal_shapes_pass_the_gradient_and_its_negation(self):
        self._assertBothVjps(
            [[1.5, -2.5], [3.0, 4.0]],
            [[0.5, 2.0], [-1.0, 8.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            expected=[[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]],
            shapes=[(2, 2), (2, 2)],
        )

    def test_a_stretched_singleton_axis_sums_the_positions_it_fed(self):
        """``(1, 3) - (2, 1)``: each operand reduces a different axis."""
        self._assertBothVjps(
            [[1.5, -2.5, 3.0]],
            [[2.0], [-4.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[5.0, 7.0, 9.0], [-6.0, -15.0]],
            shapes=[(1, 3), (2, 1)],
        )

    def test_an_added_leading_axis_is_summed_away(self):
        """``(2, 3) - (3,)``: the rank the broadcast added is reduced out."""
        self._assertBothVjps(
            [[1.5, -2.5, 3.0], [2.0, 4.0, 6.0]],
            [0.5, 2.0, -1.0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-5.0, -7.0, -9.0]],
            shapes=[(2, 3), (3,)],
        )

    def test_float32_gradients_keep_their_dtype(self):
        self._assertBothVjps(
            [[1.5, -2.5, 3.0]],
            [[2.0], [-4.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            expected=[[5.0, 7.0, 9.0], [-6.0, -15.0]],
            shapes=[(1, 3), (2, 1)],
            dtype=ts.float32,
        )

    def test_both_scalar_forms_keep_their_sign(self):
        """``x - 3`` differentiates to one; ``3 - x`` to minus one."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        seed = ts.Tensor([1.0, 2.0])
                        x = ts.Variable(ts.Tensor([1.5, -2.5]))
                        forward = ts.grad(
                            x - 3.0, x, grad_outputs=seed, create_graph=create_graph
                        )
                        y = ts.Variable(ts.Tensor([1.5, -2.5]))
                        reflected = ts.grad(
                            3.0 - y, y, grad_outputs=seed, create_graph=create_graph
                        )
                        if create_graph:
                            forward, reflected = forward.data, reflected.data
                    self.assertProduced(
                        forward, values=[1.0, 2.0], shape=(2,), backend=backend
                    )
                    self.assertProduced(
                        reflected, values=[-1.0, -2.0], shape=(2,), backend=backend
                    )

    def test_only_one_operand_may_be_requested(self):
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                for wanted in (0, 1):
                    with self.subTest(
                        backend=backend, create_graph=create_graph, input=wanted
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            operands = [
                                ts.Variable(ts.Tensor([[1.5, -2.5, 3.0]])),
                                ts.Variable(ts.Tensor([[2.0], [-4.0]])),
                            ]
                            operands[1 - wanted].requires_grad = False
                            result = ts.grad(
                                operands[0] - operands[1],
                                operands[wanted],
                                grad_outputs=ts.Tensor(
                                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                                ),
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        expected = [[5.0, 7.0, 9.0], [-6.0, -15.0]][wanted]
                        shape = [(1, 3), (2, 1)][wanted]
                        self.assertProduced(
                            produced, values=expected, shape=shape, backend=backend
                        )

    def test_a_repeated_operand_cancels_to_a_real_zero(self):
        """``(x - y) - x`` is ``-y``, so the gradient by ``x`` is zero.

        The two contributions reach the same Variable with opposite signs and
        must be accumulated, not dropped: a requested derivative whose value
        is mathematically zero is a zero, never a missing gradient.
        """
        for backend in self.BACKENDS:
            for size in self.SIZES:
                for create_graph in (False, True):
                    with self.subTest(
                        backend=backend, size=size, create_graph=create_graph
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            x = ts.Variable(ts.full((size,), 2.0, dtype=ts.float64))
                            y = ts.Variable(ts.full((2, 1), 5.0, dtype=ts.float64))
                            seed = ts.full((2, size), 1.0, dtype=ts.float64)
                            result = ts.grad(
                                (x - y) - x,
                                x,
                                grad_outputs=seed,
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        self.assertIsNotNone(produced)
                        self.assertProduced(
                            produced,
                            values=[0.0] * size,
                            shape=(size,),
                            backend=backend,
                        )

    def test_the_recorded_gradient_differentiates_by_its_upstream_seed(self):
        """The right operand's VJP is linear and negative in the seed.

        ``gb`` sums the seed along axis 1 and negates it, so differentiating
        it back with a cotangent spreads that cotangent, negated, over every
        position each sum consumed.
        """
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([[1.5, -2.5, 3.0]]))
                    b = ts.Variable(ts.Tensor([[2.0], [-4.0]]))
                    seed = ts.Variable(
                        ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    )
                    _, right = ts.grad(
                        a - b, [a, b], grad_outputs=seed, create_graph=True
                    )
                    self.assertProduced(
                        right.data,
                        values=[-6.0, -15.0],
                        shape=(2, 1),
                        backend=backend,
                    )

                    by_seed = ts.grad(
                        right, seed, grad_outputs=ts.Tensor([[2.0], [3.0]])
                    )
                self.assertProduced(
                    by_seed,
                    values=[-2.0, -2.0, -2.0, -3.0, -3.0, -3.0],
                    shape=(2, 3),
                    backend=backend,
                )

    def test_the_recorded_gradient_replays_with_a_changed_seed(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([[1.5, -2.5, 3.0]]))
                    b = ts.Variable(ts.Tensor([[2.0], [-4.0]]))
                    seed = ts.Variable(
                        ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    )
                    _, right = ts.grad(
                        a - b, [a, b], grad_outputs=seed, create_graph=True
                    )
                    program = Computation(right)
                    self.assertProduced(
                        program.forward(),
                        values=[-6.0, -15.0],
                        shape=(2, 1),
                        backend=backend,
                    )

                    seed.data = ts.Tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
                    replayed = program.forward()
                self.assertProduced(
                    replayed,
                    values=[-60.0, -150.0],
                    shape=(2, 1),
                    backend=backend,
                )

    def test_signed_zero_and_infinity_survive_the_negation(self):
        """Negation is exact, so the specified signs reach the gradient."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    a = ts.Variable(ts.Tensor([1.0, 2.0, 3.0]))
                    b = ts.Variable(ts.Tensor([1.0, 2.0, 3.0]))
                    seed = ts.Tensor([0.0, -0.0, math.inf])
                    produced = ts.grad(a - b, b, grad_outputs=seed)
                values = produced.tolist()
                self.assertEqual(math.copysign(1.0, values[0]), -1.0, "-0.0")
                self.assertEqual(math.copysign(1.0, values[1]), 1.0, "+0.0")
                self.assertEqual(values[2], -math.inf)
                self.assertEqual(produced.backend_storage.kind, backend)


class NegationVjpTests(unittest.TestCase):
    """Negation's single ``backward``, and where it runs.

    ``-x`` and the gradient of ``a - b`` with respect to ``b`` are the same
    operation, so one derivative and one execution contract serve both.
    """

    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def test_a_small_negation_stays_on_the_selected_backend(self):
        """Two elements: under the policy this path used to apply."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    produced = -ts.Tensor([1.5, -2.5], dtype=ts.float64)
                self.assertEqual(produced.tolist(), [-1.5, 2.5])
                self.assertIs(produced.dtype, ts.float64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_the_vjp_negates_the_upstream_gradient(self):
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    with ts.use_backend(backend):
                        x = ts.Variable(ts.Tensor([1.5, -2.5]))
                        result = ts.grad(
                            -x,
                            x,
                            grad_outputs=ts.Tensor([1.0, 2.0]),
                            create_graph=create_graph,
                        )
                        produced = result.data if create_graph else result
                    self.assertEqual(produced.tolist(), [-1.0, -2.0])
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_recorded_negation_differentiates_by_its_seed(self):
        """d(-seed)/d(seed) applied to a cotangent is that cotangent negated."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    x = ts.Variable(ts.Tensor([1.5, -2.5]))
                    seed = ts.Variable(ts.Tensor([1.0, 2.0]))
                    first = ts.grad(-x, x, grad_outputs=seed, create_graph=True)
                    self.assertEqual(first.data.tolist(), [-1.0, -2.0])

                    by_seed = ts.grad(
                        first, seed, grad_outputs=ts.Tensor([3.0, 5.0])
                    )
                self.assertEqual(by_seed.tolist(), [-3.0, -5.0])
                self.assertEqual(by_seed.backend_storage.kind, backend)

    def test_a_backend_that_cannot_negate_a_dtype_says_so(self):
        """CUDA keeps integers off the device, and now reports it.

        Its result conversion declines an integer dtype, which used to send
        the work to the Python reference. Removing that fallback makes the
        decline visible instead of silently relocating the computation.
        """
        from tensors.backend.config import BackendOperationUnsupportedError

        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    values = ts.Tensor([1, -2, 3], dtype=ts.int64)
                    if backend == "cuda":
                        with self.assertRaises(
                            BackendOperationUnsupportedError
                        ) as raised:
                            -values
                        self.assertIn("cuda", str(raised.exception))
                        self.assertIn("negate", str(raised.exception))
                        return
                    produced = -values
                self.assertEqual(produced.tolist(), [-1, 2, -3])
                self.assertIs(produced.dtype, ts.int64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_one_strict_entry_point_serves_both_uses(self):
        """The duplicate the subtraction VJP needed is gone."""
        import inspect

        import tensors.backend as backend_package
        from tensors.backend.dispatch.elementwise import negate as module

        self.assertFalse(hasattr(backend_package, "execute_vjp_negate"))
        source = inspect.getsource(module.execute_negate)
        for forbidden in (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "python.kernels",
            "as reference",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("validate_backend_residency", source)
        self.assertIn("load_backend", source)
        self.assertIn("BackendOperationUnsupportedError", source)


class PowerVjpTests(unittest.TestCase):
    """Power's single ``backward``, over the region table of section 12.7.2.

    Expected values are the differentiation rules: d(b**e)/db is
    ``e * b**(e-1)`` and d(b**e)/de is ``b**e * ln(b)``, with the table's
    conventions where those are not defined. Section 12.7.4 states that the
    accuracy bounds on ``**`` itself are not imposed on a whole gradient
    expression, so the comparisons below are to the rule's value at the
    precision the rule can carry, never to another backend's output.
    """

    BACKENDS = ("python", "numpy", "cuda")

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def _gradients(self, backend, base, exponent, seed, *, create_graph):
        with ts.use_backend(backend):
            b = ts.Variable(ts.Tensor(base, dtype=ts.float64), name="base")
            e = ts.Variable(ts.Tensor(exponent, dtype=ts.float64), name="exponent")
            produced = ts.grad(
                b**e,
                [b, e],
                grad_outputs=ts.Tensor(seed, dtype=ts.float64),
                create_graph=create_graph,
            )
            if create_graph:
                produced = [item.data for item in produced]
            return produced

    def test_the_ordinary_region_follows_the_differentiation_rules(self):
        """``e * b**(e-1)`` and ``b**e * ln(b)``."""
        expected_base = [3.0 * 2.0**2.0, 2.0 * 3.0**1.0]
        expected_exponent = [2.0**3.0 * math.log(2.0), 3.0**2.0 * math.log(3.0)]
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    base, exponent = self._gradients(
                        backend, [2.0, 3.0], [3.0, 2.0], [1.0, 1.0],
                        create_graph=create_graph,
                    )
                    for produced, expected in (
                        (base, expected_base),
                        (exponent, expected_exponent),
                    ):
                        for got, want in zip(produced.tolist(), expected):
                            self.assertAlmostEqual(got, want, places=12)
                        self.assertEqual(tuple(produced.shape), (2,))
                        self.assertIs(produced.dtype, ts.float64)
                        self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_negative_base_keeps_its_base_gradient_and_nans_the_exponent(self):
        """Section 12.7.3: an undefined derivative does not discard the other."""
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    base, exponent = self._gradients(
                        backend, [-2.0], [3.0], [1.0], create_graph=create_graph
                    )
                    self.assertEqual(base.tolist(), [12.0])
                    self.assertTrue(math.isnan(exponent.tolist()[0]))
                    self.assertEqual(base.backend_storage.kind, backend)

    def test_the_zero_and_unit_regions(self):
        cases = (
            # base, exponent, seed, base gradient, exponent gradient
            ([0.0], [3.0], [1.0], 0.0, 0.0),
            ([0.0], [1.0], [1.0], 1.0, 0.0),
            ([2.0], [0.0], [1.0], 0.0, math.log(2.0)),
            ([1.0], [5.0], [1.0], 5.0, 0.0),
        )
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                for b, e, seed, want_base, want_exp in cases:
                    with self.subTest(
                        backend=backend, create_graph=create_graph, base=b[0],
                        exponent=e[0],
                    ):
                        self._require(backend)
                        reset_graph_state()
                        base, exponent = self._gradients(
                            backend, b, e, seed, create_graph=create_graph
                        )
                        self.assertAlmostEqual(base.tolist()[0], want_base, places=12)
                        self.assertAlmostEqual(
                            exponent.tolist()[0], want_exp, places=12
                        )

    def test_the_range_safe_regions_survive_both_reverse_modes(self):
        """Intermediates that overflow or underflow; the result does not.

        ``1e-308 ** 2`` scaled by ``1e308`` is exactly 2, and
        ``1e308 ** -1`` scaled by ``1e308`` is exactly ``-1e-308``. A naive
        product forms an intermediate outside the range on the way to each.
        """
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(case="small base", backend=backend,
                                  create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    base, _ = self._gradients(
                        backend, [1.0e-308], [2.0], [1.0e308],
                        create_graph=create_graph,
                    )
                    self.assertTrue(
                        math.isclose(base.tolist()[0], 2.0, rel_tol=1.0e-12)
                    )
                    self.assertEqual(base.backend_storage.kind, backend)

                with self.subTest(case="large base", backend=backend,
                                  create_graph=create_graph):
                    reset_graph_state()
                    base, _ = self._gradients(
                        backend, [1.0e308], [-1.0], [1.0e308],
                        create_graph=create_graph,
                    )
                    self.assertTrue(
                        math.isclose(base.tolist()[0], -1.0e-308, rel_tol=1.0e-12)
                    )

    def test_a_broadcast_gradient_reduces_to_each_operand(self):
        """``(1, 3) ** (2, 1)``: each operand reduces the other's axis."""
        bases = [2.0, 3.0, 4.0]
        exponents = [2.0, 3.0]
        expected_base = [
            sum(e * b ** (e - 1.0) for e in exponents) for b in bases
        ]
        expected_exponent = [
            sum(b**e * math.log(b) for b in bases) for e in exponents
        ]
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                with self.subTest(backend=backend, create_graph=create_graph):
                    self._require(backend)
                    reset_graph_state()
                    base, exponent = self._gradients(
                        backend,
                        [bases],
                        [[e] for e in exponents],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        create_graph=create_graph,
                    )
                    self.assertEqual(tuple(base.shape), (1, 3))
                    self.assertEqual(tuple(exponent.shape), (2, 1))
                    for got, want in zip(base.tolist(), expected_base):
                        self.assertAlmostEqual(got, want, places=10)
                    for got, want in zip(exponent.tolist(), expected_exponent):
                        self.assertAlmostEqual(got, want, places=10)
                    self.assertEqual(base.backend_storage.kind, backend)
                    self.assertEqual(exponent.backend_storage.kind, backend)

    def test_either_operand_may_be_requested_alone(self):
        for backend in self.BACKENDS:
            for create_graph in (False, True):
                for wanted in (0, 1):
                    with self.subTest(
                        backend=backend, create_graph=create_graph, input=wanted
                    ):
                        self._require(backend)
                        reset_graph_state()
                        with ts.use_backend(backend):
                            operands = [
                                ts.Variable(ts.Tensor([2.0], dtype=ts.float64)),
                                ts.Variable(ts.Tensor([3.0], dtype=ts.float64)),
                            ]
                            operands[1 - wanted].requires_grad = False
                            result = ts.grad(
                                operands[0] ** operands[1],
                                operands[wanted],
                                grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                                create_graph=create_graph,
                            )
                            produced = result.data if create_graph else result
                        expected = [12.0, 8.0 * math.log(2.0)][wanted]
                        self.assertAlmostEqual(
                            produced.tolist()[0], expected, places=12
                        )
                        self.assertEqual(produced.backend_storage.kind, backend)

    def test_a_second_derivative_is_available_numerically(self):
        """d2(x**3)/dx2 is 6x, from the recorded first derivative."""
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    x = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
                    ones = ts.Tensor([1.0], dtype=ts.float64)
                    first = ts.grad(x**3.0, x, grad_outputs=ones, create_graph=True)
                    second = ts.grad(first, x, grad_outputs=ones)
                self.assertEqual(first.data.tolist(), [12.0])  # 3x^2
                self.assertEqual(second.tolist(), [12.0])  # 6x
                self.assertEqual(second.backend_storage.kind, backend)

    def test_the_recorded_gradient_replays_with_a_changed_base(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    x = ts.Variable(ts.Tensor([2.0], dtype=ts.float64))
                    first = ts.grad(
                        x**3.0,
                        x,
                        grad_outputs=ts.Tensor([1.0], dtype=ts.float64),
                        create_graph=True,
                    )
                    program = Computation(first)
                    self.assertEqual(program.forward().tolist(), [12.0])

                    x.data = ts.Tensor([5.0], dtype=ts.float64)
                    replayed = program.forward()
                self.assertEqual(replayed.tolist(), [75.0])  # 3x^2 at x = 5
                self.assertEqual(replayed.backend_storage.kind, backend)

    def test_the_gradient_primitives_do_not_restate_their_own_kernel(self):
        """Each one's first slot is its own forward, not a copy of the kernel.

        ``d(base gradient)/d(upstream gradient)`` is the base gradient, so the
        host loop that recomputed it held a second implementation of the
        section 12.7.2 table — the Python backend kernel already holds one —
        and computed it in Python whatever backend was selected.
        """
        import importlib
        import inspect

        module = importlib.import_module("tensors.operations.arithmetic.power")
        kernel = importlib.import_module(
            "tensors.backend.python.kernels.elementwise.power_base_gradient"
        )
        # The table's per-element rules belong to the kernels alone.
        for name in ("_base_gradient_value", "_exponent_gradient_value"):
            with self.subTest(helper=name):
                self.assertFalse(hasattr(module, name))
                self.assertTrue(hasattr(kernel, "_base_gradient_value"))

        for operation in (module.PowerBaseGradient, module.PowerExponentGradient):
            with self.subTest(operation=operation.name):
                source = inspect.getsource(operation.backward)
                self.assertIn("self.forward(", source)

    def test_the_mixed_second_partial_is_stated_once(self):
        """Both operations reach the same statement of it."""
        import importlib
        import inspect

        module = importlib.import_module("tensors.operations.arithmetic.power")
        for operation in (module.PowerBaseGradient, module.PowerExponentGradient):
            with self.subTest(operation=operation.name):
                source = inspect.getsource(operation.backward)
                self.assertIn("_mixed_power_derivative(", source)

    def test_the_gradient_primitives_do_not_expand_through_the_host(self):
        """Each backend's kernel broadcasts; expanding first read operands back."""
        import ast
        import inspect
        import textwrap

        import importlib

        module = importlib.import_module("tensors.operations.arithmetic.power")

        for operation in (module.PowerBaseGradient, module.PowerExponentGradient):
            with self.subTest(operation=operation.name):
                source = textwrap.dedent(inspect.getsource(operation.forward))
                self.assertNotIn("_expanded_power_inputs", source)
                host_reads = [
                    node
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Attribute) and node.attr == "_data"
                ]
                self.assertEqual(host_reads, [])


if __name__ == "__main__":
    unittest.main()
