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
        from tensors.operations.manipulation.reshape import Reshape
        from tensors.operations.reductions.sum import Sum
        from tensors.ops import Add

        for operation in (Add, Sum, Reshape):
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


if __name__ == "__main__":
    unittest.main()
