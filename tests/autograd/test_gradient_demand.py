"""Reverse gradient demand belongs to Computation, not to Operation."""

import math
import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.ops import Operation
from tensors.graph.state import reset_graph_state
from tensors.operations.losses.binary_cross_entropy import BinaryCrossEntropy
from tensors.operations.losses.cross_entropy import CrossEntropy
from tensors.operations.selection import Maximum, Minimum
from tensors.operations.selection.where import Where
from tensors.ops import Add, Div, Mul, Pow, Sub
from tensors.operations.arithmetic.divide import DivisionDenominatorGradient
from tensors.operations.arithmetic.power import PowerBaseGradient, PowerExponentGradient


class _Recorder:
    """Record the demand each VJP call receives without changing results."""

    def __init__(self, *operations):
        self.operations = operations
        self.calls: list[tuple[str, tuple[bool, ...]]] = []
        self._originals: dict = {}

    def __enter__(self):
        for operation in self.operations:
            self._originals[operation] = operation.backward
            self.calls.clear()

            def patched(
                inner,
                gradient,
                *inputs,
                needs_input_grad,
                _operation=operation,
            ):
                self.calls.append((_operation.name, needs_input_grad))
                return self._originals[_operation](
                    inner,
                    gradient,
                    *inputs,
                    needs_input_grad=needs_input_grad,
                )

            operation.backward = patched
        return self

    def __exit__(self, *exception):
        for operation, original in self._originals.items():
            operation.backward = original
        return False

    def demand_for(self, name: str) -> list[tuple[bool, ...]]:
        return [mask for label, mask in self.calls if label == name]


class OperationConfigurationTests(unittest.TestCase):
    def test_operations_carry_no_differentiation_demand(self):
        for operation in (
            Add(),
            Sub(),
            Mul(),
            Div(),
            Pow(),
            PowerBaseGradient(),
            PowerExponentGradient(),
            DivisionDenominatorGradient(),
            Where(),
            Maximum(),
            Minimum(),
        ):
            with self.subTest(operation=operation.name):
                self.assertEqual(type(operation).__slots__, ())

    def test_no_operation_configuration_encodes_reverse_demand(self):
        """Configuration must describe mathematics, never gradient demand."""
        forbidden = ("differentiate", "needs_input_grad", "requires_grad")
        seen = 0
        stack = [Operation]
        while stack:
            base = stack.pop()
            for subclass in base.__subclasses__():
                stack.append(subclass)
                seen += 1
                for name in getattr(subclass, "__slots__", ()):
                    with self.subTest(operation=subclass.__name__, slot=name):
                        for marker in forbidden:
                            self.assertNotIn(marker, name)
        self.assertGreater(seen, 40)

    def test_configured_operations_keep_their_mathematical_settings(self):
        from tensors.operations.reductions.sum import Sum

        operation = Sum(axis=(1,), keepdims=True)
        self.assertEqual(operation.axis, (1,))
        self.assertTrue(operation.keepdims)
        self.assertEqual(
            BinaryCrossEntropy(from_logits=True, reduction="sum").from_logits,
            True,
        )
        self.assertEqual(CrossEntropy(axis=1, reduction="mean").axis, 1)


class DemandContractTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_vjp_methods_receive_explicit_demand(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        output = left * right

        with _Recorder(Mul) as recorder:
            ts.grad(output, left)

        self.assertEqual(recorder.demand_for("mul"), [(True, False)])

    def test_unrequested_vjp_returns_none(self):
        left = ts.Variable([2.0])
        right = ts.Variable([3.0])

        results = Mul().backward(
            ts.Tensor([1.0]),
            left.data,
            right.data,
            needs_input_grad=(True, False),
        )

        self.assertIsInstance(results[0], ts.Tensor)
        self.assertIsNone(results[1])

    def test_requested_zero_vjp_returns_a_real_zero(self):
        """A zero derivative is a value, not an absent one."""
        base = ts.Variable([2.0])
        results = Pow().backward(
            ts.Tensor([1.0]),
            base.data,
            ts.Tensor([0.0]),
            needs_input_grad=(True, False),
        )

        self.assertIsInstance(results[0], ts.Tensor)
        self.assertEqual(results[0].tolist(), [0.0])
        self.assertIsNone(results[1])

    def test_computation_rejects_a_value_for_an_unrequested_input(self):
        class Overeager(Operation):
            name = "overeager"

            def forward(self, left, right):
                return left + right

            def backward(self, gradient, left, right, *, needs_input_grad):
                return [gradient, gradient]

        left = ts.Variable([2.0])
        right = ts.Variable([3.0], requires_grad=False)
        operation = Overeager()
        output = ts.Variable._apply_operation(operation, (left, right))

        with self.assertRaisesRegex(RuntimeError, "did not request"):
            ts.grad(output, left)

    def test_computation_rejects_none_for_a_requested_input(self):
        class Forgetful(Operation):
            name = "forgetful"

            def forward(self, left, right):
                return left + right

            def backward(self, gradient, left, right, *, needs_input_grad):
                return [None, gradient]

        left = ts.Variable([2.0])
        right = ts.Variable([3.0])
        operation = Forgetful()
        output = ts.Variable._apply_operation(operation, (left, right))

        with self.assertRaisesRegex(RuntimeError, "requested"):
            ts.grad(output, left)


class ReverseDemandPlanningTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def _expression(self):
        a = ts.Variable([2.0], name="a")
        b = ts.Variable([3.0], name="b")
        c = ts.Variable([4.0], name="c")
        return a, b, c, (a * b) + c

    def test_targeted_grad_skips_unrelated_branches(self):
        a, b, c, y = self._expression()

        with _Recorder(Add, Mul) as recorder:
            result = ts.grad(y, a)

        self.assertEqual(result.tolist(), [3.0])
        self.assertEqual(recorder.demand_for("add"), [(True, False)])
        self.assertEqual(recorder.demand_for("mul"), [(True, False)])

    def test_targeted_grad_skips_a_whole_operation(self):
        a, b, c, y = self._expression()

        with _Recorder(Add, Mul) as recorder:
            result = ts.grad(y, c)

        self.assertEqual(result.tolist(), [1.0])
        self.assertEqual(recorder.demand_for("add"), [(False, True)])
        # Nothing behind the product is wanted, so its VJP never runs.
        self.assertEqual(recorder.demand_for("mul"), [])

    def test_multiple_requested_inputs_widen_the_demand(self):
        a, b, c, y = self._expression()

        with _Recorder(Add, Mul) as recorder:
            first, second = ts.grad(y, (a, c))

        self.assertEqual(first.tolist(), [3.0])
        self.assertEqual(second.tolist(), [1.0])
        self.assertEqual(recorder.demand_for("add"), [(True, True)])
        self.assertEqual(recorder.demand_for("mul"), [(True, False)])

    def test_backward_publishes_to_every_reachable_variable(self):
        a, b, c, y = self._expression()

        with _Recorder(Add, Mul) as recorder:
            ts.backward(y)

        self.assertEqual(a.grad.tolist(), [3.0])
        self.assertEqual(b.grad.tolist(), [2.0])
        self.assertEqual(c.grad.tolist(), [1.0])
        self.assertEqual(recorder.demand_for("add"), [(True, True)])
        self.assertEqual(recorder.demand_for("mul"), [(True, True)])

    def test_frozen_operand_is_never_requested(self):
        value = ts.Variable([2.0])
        frozen = ts.Variable([3.0], requires_grad=False)
        output = value * frozen

        with _Recorder(Mul) as recorder:
            ts.backward(output)

        self.assertEqual(recorder.demand_for("mul"), [(True, False)])

    def test_create_graph_uses_the_same_demand_model(self):
        a, b, c, y = self._expression()

        with _Recorder(Add, Mul) as recorder:
            first = ts.grad(y, a, create_graph=True)

        self.assertEqual(first.data.tolist(), [3.0])
        # One ``backward`` per operation serves both reverse passes, so the
        # method the recorder patches is the one graph building calls, and it
        # carries the demand a numerical pass would carry. ``Mul`` still
        # answers with Tensors when it is handed Variables, so this currently
        # fails on the product rather than on the demand it asserts.
        self.assertEqual(
            recorder.calls, [("add", (True, False)), ("mul", (True, False))]
        )
        self.assertEqual(ts.grad(first, b).tolist(), [1.0])

    def test_one_backward_serves_both_reverse_modes(self):
        """The method a value pass calls is the one graph building calls.

        Addition defines its derivative once. Patching ``Add.backward`` is
        therefore the whole observation: if graph building reached a second
        method, the recorder would see nothing in the recorded pass, and if
        the two disagreed, the gradients would differ.
        """
        self.assertFalse(hasattr(Add, "backward_graph"))

        def reverse(create_graph):
            reset_graph_state()
            left = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]]), name="left")
            right = ts.Variable(ts.Tensor([[10.0], [20.0]]), name="right")
            seed = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            with _Recorder(Add) as recorder:
                first, second = ts.grad(
                    left + right,
                    [left, right],
                    grad_outputs=seed,
                    create_graph=create_graph,
                )
            if create_graph:
                first, second = first.data, second.data
            return recorder.calls, [first.tolist(), second.tolist()]

        numerical_calls, numerical_values = reverse(False)
        recorded_calls, recorded_values = reverse(True)

        self.assertEqual(numerical_calls, [("add", (True, True))])
        self.assertEqual(recorded_calls, numerical_calls)
        self.assertEqual(numerical_values, [[5.0, 7.0, 9.0], [6.0, 15.0]])
        self.assertEqual(recorded_values, numerical_values)

    def test_an_unrequested_addition_operand_is_skipped_in_both_modes(self):
        """Demand prunes the recorded reverse pass as it prunes a value pass."""
        for create_graph in (False, True):
            with self.subTest(create_graph=create_graph):
                reset_graph_state()
                wanted = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]]), name="wanted")
                frozen = ts.Variable(
                    ts.Tensor([[10.0], [20.0]]),
                    name="frozen",
                    requires_grad=False,
                )
                seed = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                with _Recorder(Add) as recorder:
                    result = ts.grad(
                        wanted + frozen,
                        wanted,
                        grad_outputs=seed,
                        create_graph=create_graph,
                    )
                produced = result.data if create_graph else result
                self.assertEqual(recorder.demand_for("add"), [(True, False)])
                self.assertEqual(produced.tolist(), [5.0, 7.0, 9.0])
                self.assertEqual(tuple(produced.shape), (1, 3))

    def test_demand_follows_a_requires_grad_change_after_replay(self):
        value = ts.Variable([2.0], name="value")
        weight = ts.Variable([3.0], name="weight")
        output = value * weight
        computation = Computation(output)

        self.assertEqual(ts.grad(output, value).tolist(), [3.0])

        value.requires_grad = False
        with self.assertRaisesRegex(RuntimeError, "modified after its forward"):
            ts.grad(output, weight)

        computation.forward()
        self.assertEqual(ts.grad(output, weight).tolist(), [2.0])
        self.assertIsNone(ts.grad(output, value))

        # Demand is resolved per reverse call, so the live slots reflect the
        # Variables' current state rather than the state traced with.
        value_slot = computation._node_slots[value.node]
        weight_slot = computation._node_slots[weight.node]
        self.assertEqual(
            computation._live_slots((weight,)) & {value_slot, weight_slot},
            {weight_slot},
        )
        self.assertNotIn(value_slot, computation._live_slots(None))

        value.requires_grad = True
        computation.forward()
        self.assertEqual(
            computation._live_slots(None) & {value_slot, weight_slot},
            {value_slot, weight_slot},
        )

    def test_jacobian_and_hessian_remain_correct(self):
        value = ts.Variable([2.0, 3.0])
        output = ts.concat([value[0] ** 2.0, value[0] * value[1]])

        jacobian = ts.jacobian(output, value)
        self.assertEqual(jacobian.tolist(), [4.0, 0.0, 3.0, 2.0])

        scalar = ts.sum(value[0] ** 2.0 + value[0] * value[1])
        hessian = ts.hessian(scalar, value)
        self.assertEqual(hessian.tolist(), [2.0, 1.0, 1.0, 0.0])


class DemandScopedDomainTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_an_undefined_power_derivative_is_nan_not_an_exception(self):
        """Section 12.7, rules G1 and G2.

        This test previously required ``ValueError`` from a requested
        exponent derivative at a negative base, and required ``ts.backward``
        to raise as well. Section 12.7.2 classifies that derivative as NaN —
        ``ln x`` is undefined for ``x < 0`` — and rule G2 states that
        differentiation does not raise on a numerical condition. Rule G1
        adds that the condition must not suppress the base derivative, which
        the old behaviour did: ``ts.backward`` raised and both gradients were
        lost, including the valid one.
        """
        base = ts.Variable([-2.0])
        exponent = ts.Variable([2.0])
        output = base**exponent

        # The base derivative exists at a negative base with an integral
        # exponent, and is returned whether or not the other is requested.
        self.assertEqual(ts.grad(output, base).tolist(), [-4.0])

        undefined = ts.grad(output, exponent).tolist()
        self.assertTrue(math.isnan(undefined[0]), undefined)

        ts.backward(output)
        self.assertEqual(base.grad.tolist(), [-4.0])
        self.assertTrue(math.isnan(exponent.grad.tolist()[0]))

    def test_binary_cross_entropy_higher_order_domain_follows_demand(self):
        prediction = ts.Variable([0.0])
        target = ts.Variable([0.0])
        loss = ts.binary_cross_entropy(prediction, target)

        # A boundary probability has no higher-order target derivative, but
        # the prediction derivative remains available there.
        first = ts.grad(loss, prediction, create_graph=True)
        self.assertEqual(first.data.tolist(), [1.0])

        with self.assertRaisesRegex(ValueError, "strictly\\s+between 0 and 1"):
            ts.grad(loss, target, create_graph=True)


class FusionDemandTests(unittest.TestCase):
    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @unittest.skipUnless(
        "cuda" in ts.available_backends(),
        "requires an available CUDA backend",
    )
    def test_forward_fusion_is_independent_of_backward_demand(self):
        from unittest.mock import patch

        from tensors.backend.cuda import kernels as cuda_backend
        from tensors.backend import _clear_backend_kernel_cache

        def expression(base, exponent):
            return ts.sin(base**exponent) * 2.0 + 1.0

        with ts.use_backend("cuda"):
            base = ts.Variable(ts.full((4_096,), 1.5))
            exponent = ts.Variable(ts.full((4_096,), 2.0))
            output = expression(base, exponent)
            computation = Computation(output)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fusion:
                _clear_backend_kernel_cache()
                computation.forward()

        # A differentiable exponent no longer prevents the forward fusion.
        fusion.assert_called_once()

    @unittest.skipUnless(
        "cuda" in ts.available_backends(),
        "requires an available CUDA backend",
    )
    def test_fused_backward_falls_back_for_an_unsupported_derivative(self):
        def expression(base, exponent):
            return ts.sin(base**exponent) * 2.0 + 1.0

        with ts.use_backend("python"):
            reference_base = ts.Variable(ts.full((4_096,), 1.5))
            reference_exponent = ts.Variable(ts.full((4_096,), 2.0))
            reference = expression(reference_base, reference_exponent)
            expected_base, expected_exponent = ts.grad(
                reference,
                (reference_base, reference_exponent),
                ts.ones((4_096,)),
            )

        with ts.use_backend("cuda"):
            base = ts.Variable(ts.full((4_096,), 1.5))
            exponent = ts.Variable(ts.full((4_096,), 2.0))
            output = expression(base, exponent)
            # The fused VJP carries no external power derivative, so the group
            # falls back rather than reading a row the kernel never wrote.
            actual_base, actual_exponent = ts.grad(
                output,
                (base, exponent),
                ts.ones((4_096,)),
            )

        self.assertAlmostEqual(
            actual_base.tolist()[0], expected_base.tolist()[0], places=10
        )
        self.assertAlmostEqual(
            actual_exponent.tolist()[0],
            expected_exponent.tolist()[0],
            places=10,
        )

    @unittest.skipUnless(
        "cuda" in ts.available_backends(),
        "requires an available CUDA backend",
    )
    def test_fused_backward_still_serves_a_supported_demand(self):
        with ts.use_backend("python"):
            reference_value = ts.Variable(ts.full((4_096,), 1.5))
            expected = ts.grad(
                ts.sin(reference_value**2.0) * 2.0 + 1.0,
                reference_value,
                ts.ones((4_096,)),
            )

        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 1.5))
            output = ts.sin(value**2.0) * 2.0 + 1.0
            actual = ts.grad(output, value, ts.ones((4_096,)))

        self.assertAlmostEqual(actual.tolist()[0], expected.tolist()[0], places=10)


if __name__ == "__main__":
    unittest.main()
