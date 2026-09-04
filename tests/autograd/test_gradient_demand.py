"""Reverse gradient demand belongs to Computation, not to Operation."""

import unittest

import tensors as ts
from tensors.graph import Computation
from tensors.ops import Operation
from tensors.graph.state import reset_graph_state
from tensors.math.binary_cross_entropy import BinaryCrossEntropy
from tensors.math.cross_entropy import CrossEntropy
from tensors.math.elementwise_extrema import Maximum, Minimum
from tensors.math.where import Where
from tensors.ops import Add, Div, Mul, Pow, Sub
from tensors.ops.div import DivisionDenominatorGradient
from tensors.ops.pow import PowerBaseGradient, PowerExponentGradient


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
        from tensors.math.sum import Sum

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
        output = ts.Variable._from_operation(
            operation.forward(left.data, right.data),
            operation,
            (left, right),
        )

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
        output = ts.Variable._from_operation(
            operation.forward(left.data, right.data),
            operation,
            (left, right),
        )

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
        # backward_graph carries the same demand; the numerical VJP is unused.
        self.assertEqual(recorder.calls, [])
        self.assertEqual(ts.grad(first, b).tolist(), [1.0])

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
        value_slot = computation._variable_slots[value]
        weight_slot = computation._variable_slots[weight]
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

    def test_power_domain_check_only_guards_a_requested_derivative(self):
        base = ts.Variable([-2.0])
        exponent = ts.Variable([2.0])
        output = base ** exponent

        # The base derivative is defined at a negative base.
        self.assertEqual(ts.grad(output, base).tolist(), [-4.0])

        # The exponent derivative is not, and only its request raises.
        with self.assertRaisesRegex(ValueError, "non-negative bases"):
            ts.grad(output, exponent)
        with self.assertRaisesRegex(ValueError, "non-negative bases"):
            ts.backward(output)

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

        from tensors.backend import cuda as cuda_backend
        from tensors.backend import _clear_backend_kernel_cache

        def expression(base, exponent):
            return ts.sin(base ** exponent) * 2.0 + 1.0

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
            return ts.sin(base ** exponent) * 2.0 + 1.0

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

        self.assertAlmostEqual(actual_base[0], expected_base[0], places=10)
        self.assertAlmostEqual(
            actual_exponent[0],
            expected_exponent[0],
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
                ts.sin(reference_value ** 2.0) * 2.0 + 1.0,
                reference_value,
                ts.ones((4_096,)),
            )

        with ts.use_backend("cuda"):
            value = ts.Variable(ts.full((4_096,), 1.5))
            output = ts.sin(value ** 2.0) * 2.0 + 1.0
            actual = ts.grad(output, value, ts.ones((4_096,)))

        self.assertAlmostEqual(actual[0], expected[0], places=10)


if __name__ == "__main__":
    unittest.main()
