"""The second derivatives of ``**``, on the backend the selection names.

`docs/arithmetic-semantics.md` section 12.7 states power's first derivatives
and the rules they answer to. Those rules are not exhausted by the first
order: a second derivative that raises on a numerical condition breaks G2, one
that reads an operand to decide a region breaks G3, and one computed in a host
loop breaks G6 whatever backend was selected. Differentiating ``PowerBaseVJP``
and ``PowerExponentVJP`` used to do all three.

The expectations here come from the derivatives themselves,

.. math:: f_{bb} = e(e-1)b^{e-2}, \\quad
          f_{be} = b^{e-1}(1 + e\\ln b), \\quad
          f_{ee} = b^{e}(\\ln b)^2

and from the range and domain rules those inherit, never from another
backend's output.
"""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state
from tensors.operations.arithmetic.power import (
    PowerBaseBaseVJP,
    PowerBaseVJP,
    PowerExponentExponentVJP,
    PowerExponentVJP,
    PowerMixedVJP,
)

BACKENDS = ("python", "numpy", "cuda")


class PowerSecondDerivativeTestCase(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    self.skipTest(f"the {backend} backend is not available here")
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def second(self, base_values, exponent_values, wrt, of):
        """``d/d<wrt>`` of ``d(sum(b ** e))/d<of>``, and where it lives."""
        base = ts.Variable(ts.Tensor(base_values, dtype=ts.float64), requires_grad=True)
        exponent = ts.Variable(
            ts.Tensor(exponent_values, dtype=ts.float64), requires_grad=True
        )
        first = {"base": base, "exponent": exponent}[of]
        (gradient,) = ts.grad(ts.sum(base**exponent), (first,), create_graph=True)
        wanted = {"base": base, "exponent": exponent}[wrt]
        (second,) = ts.grad(ts.sum(gradient), (wanted,))
        return second


class TheOrdinaryRegion(PowerSecondDerivativeTestCase):
    """Where every partial exists, each is the formula."""

    def test_the_second_partial_by_the_base(self):
        def body(backend):
            produced = self.second([2.0, 3.0, 0.5], [3.0, 2.0, 4.0], "base", "base")
            expected = [
                exponent * (exponent - 1.0) * base ** (exponent - 2.0)
                for base, exponent in ((2.0, 3.0), (3.0, 2.0), (0.5, 4.0))
            ]
            for produced_value, expected_value in zip(produced.tolist(), expected):
                self.assertAlmostEqual(produced_value, expected_value, places=10)
            self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_the_second_partial_by_the_exponent(self):
        def body(backend):
            produced = self.second(
                [2.0, 3.0, 0.5], [3.0, 2.0, 4.0], "exponent", "exponent"
            )
            expected = [
                base**exponent * math.log(base) ** 2
                for base, exponent in ((2.0, 3.0), (3.0, 2.0), (0.5, 4.0))
            ]
            for produced_value, expected_value in zip(produced.tolist(), expected):
                self.assertAlmostEqual(produced_value, expected_value, places=10)
            self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_the_mixed_partial_is_the_same_from_either_side(self):
        """``d2f/db de`` and ``d2f/de db`` are one value, and one rule."""

        def body(backend):
            from_base = self.second([2.0, 3.0], [3.0, 2.0], "exponent", "base")
            from_exponent = self.second([2.0, 3.0], [3.0, 2.0], "base", "exponent")
            expected = [
                base ** (exponent - 1.0) * (1.0 + exponent * math.log(base))
                for base, exponent in ((2.0, 3.0), (3.0, 2.0))
            ]
            for produced_value, expected_value in zip(from_base.tolist(), expected):
                self.assertAlmostEqual(produced_value, expected_value, places=10)
            for produced_value, expected_value in zip(from_exponent.tolist(), expected):
                self.assertAlmostEqual(produced_value, expected_value, places=10)
            self.assertEqual(from_base.backend_storage.kind, backend)
            self.assertEqual(from_exponent.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_an_exponent_of_zero_or_one_has_no_base_curvature(self):
        """``b ** 0`` is constant and ``b ** 1`` is linear, so both give zero.

        The formula gives ``0 * inf`` at a zero base here. An exactly zero
        factor is an exact zero, which is what makes these rows right rather
        than NaN.
        """

        def body(backend):
            produced = self.second([2.0, 5.0, 0.0], [0.0, 1.0, 1.0], "base", "base")
            self.assertEqual(produced.tolist(), [0.0, 0.0, 0.0])
            self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)


class TheDomainRules(PowerSecondDerivativeTestCase):
    """G1 and G2 at the second order: NaN, never an exception, never both lost."""

    def test_a_negative_base_keeps_the_base_curvature_it_has(self):
        """``e - 2`` is integral too, so the power is real and the value exists.

        The exponent partials do not exist, because ``ln x`` does not. This
        computation used to raise on the logarithm and lose all three.
        """

        def body(backend):
            produced = self.second([-2.0], [3.0], "base", "base")
            self.assertEqual(produced.tolist(), [3.0 * 2.0 * (-2.0)])
            mixed = self.second([-2.0], [3.0], "exponent", "base")
            self.assertTrue(math.isnan(mixed.tolist()[0]))
            self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_a_negative_base_with_a_non_integral_exponent_is_nan_throughout(self):
        def body(backend):
            for wrt, of in (("base", "base"), ("exponent", "exponent")):
                produced = self.second([-2.0], [2.5], wrt, of)
                self.assertTrue(math.isnan(produced.tolist()[0]))

        self.for_each_backend(body)

    def test_a_zero_base_below_the_constant_exponents_does_not_raise(self):
        """It used to. An absent derivative is NaN, and a divergence is signed.

        ``d2(b ** 0.5)/db2`` is ``-0.25 b ** -1.5``, which diverges to minus
        infinity as the base approaches zero; the mixed partial has no limit
        there at all.
        """

        def body(backend):
            curvature = self.second([0.0], [0.5], "base", "base")
            self.assertEqual(curvature.tolist(), [-math.inf])
            mixed = self.second([0.0], [0.5], "exponent", "base")
            self.assertTrue(math.isnan(mixed.tolist()[0]))

        self.for_each_backend(body)

    def test_a_zero_base_with_a_large_exponent_is_flat_in_the_exponent(self):
        """``f(0, e)`` is zero for every positive exponent, so it is constant."""

        def body(backend):
            self.assertEqual(
                self.second([0.0], [3.0], "exponent", "exponent").tolist(), [0.0]
            )
            self.assertEqual(
                self.second([0.0], [3.0], "exponent", "base").tolist(), [0.0]
            )

        self.for_each_backend(body)

    def test_a_requested_partial_survives_an_absent_one(self):
        """Rule G1 at the second order, asked of the operation directly."""

        def body(backend):
            outer = ts.Tensor([1.0], dtype=ts.float64)
            grad = ts.Tensor([1.0], dtype=ts.float64)
            base = ts.Tensor([-2.0], dtype=ts.float64)
            exponent = ts.Tensor([3.0], dtype=ts.float64)
            produced = PowerBaseVJP().backward(
                outer, grad, base, exponent, needs_input_grad=(False, True, True)
            )
            self.assertIsNone(produced[0])
            self.assertEqual(produced[1].tolist(), [-12.0])
            self.assertTrue(math.isnan(produced[2].tolist()[0]))

        self.for_each_backend(body)


class TheRange(PowerSecondDerivativeTestCase):
    """A power that leaves the range must not take the result with it."""

    def test_a_curvature_survives_an_overflowing_power(self):
        """``1e200 ** 2`` is not representable; ``2 * 1 * 1e200 ** 0`` is."""

        def body(backend):
            produced = self.second([1e200], [2.0], "base", "base")
            self.assertEqual(produced.tolist(), [2.0])

        self.for_each_backend(body)

    def test_a_curvature_survives_an_underflowing_power(self):
        def body(backend):
            produced = self.second([1e-200], [3.0], "base", "base")
            self.assertEqual(produced.tolist(), [6e-200])

        self.for_each_backend(body)

    def test_a_large_curvature_is_reached_rather_than_rounded_away(self):
        """``3 * 2 * 1e300`` is representable and is the answer."""

        def body(backend):
            produced = self.second([1e300], [3.0], "base", "base")
            self.assertEqual(produced.tolist(), [6e300])

        self.for_each_backend(body)


class BroadcastingAndShapes(PowerSecondDerivativeTestCase):
    def test_a_broadcast_second_derivative_reduces_to_each_operand(self):
        def body(backend):
            base = ts.Variable(
                ts.Tensor([[2.0, 3.0]], dtype=ts.float64), requires_grad=True
            )
            exponent = ts.Variable(
                ts.Tensor([[2.0], [3.0]], dtype=ts.float64), requires_grad=True
            )
            (gradient,) = ts.grad(ts.sum(base**exponent), (base,), create_graph=True)
            curvature, mixed = ts.grad(ts.sum(gradient), (base, exponent))
            self.assertEqual(tuple(curvature.shape), (1, 2))
            self.assertEqual(tuple(mixed.shape), (2, 1))
            # Column j of the base curvature sums over both exponents.
            expected = [
                sum(
                    exponent_value
                    * (exponent_value - 1.0)
                    * base_value ** (exponent_value - 2.0)
                    for exponent_value in (2.0, 3.0)
                )
                for base_value in (2.0, 3.0)
            ]
            for produced, wanted in zip(curvature.tolist(), expected):
                self.assertAlmostEqual(produced, wanted, places=10)
            self.assertEqual(curvature.backend_storage.kind, backend)
            self.assertEqual(mixed.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_each_partial_carries_its_own_operand_dtype(self):
        """Rule G5 at the second order."""

        def body(backend):
            outer = ts.Tensor([1.0], dtype=ts.float64)
            grad = ts.Tensor([1.0], dtype=ts.float64)
            base = ts.Tensor([2.0], dtype=ts.float64)
            exponent = ts.Tensor([3.0], dtype=ts.float32)
            produced = PowerBaseVJP().backward(
                outer, grad, base, exponent, needs_input_grad=(False, True, True)
            )
            self.assertIs(produced[1].dtype, ts.float64)
            self.assertIs(produced[2].dtype, ts.float32)

        self.for_each_backend(body)


class WhereTheSecondDerivativesExecute(PowerSecondDerivativeTestCase):
    """Rule G6 at the second order: the selection decides, at every size."""

    def test_every_size_stays_on_the_selected_backend(self):
        def body(backend):
            for size in (1, 3, 31, 4096):
                base = ts.Variable(
                    ts.Tensor([2.0] * size, dtype=ts.float64), requires_grad=True
                )
                exponent = ts.Variable(
                    ts.Tensor([3.0] * size, dtype=ts.float64), requires_grad=True
                )
                (gradient,) = ts.grad(
                    ts.sum(base**exponent), (base,), create_graph=True
                )
                curvature, mixed = ts.grad(ts.sum(gradient), (base, exponent))
                self.assertEqual(curvature.backend_storage.kind, backend)
                self.assertEqual(mixed.backend_storage.kind, backend)
                self.assertEqual(curvature.tolist()[0], 12.0)

        self.for_each_backend(body)

    def test_the_second_partials_are_operations_with_their_own_kernels(self):
        """They are primitives because the grouping cannot be assembled."""
        from tensors.ops import Operation

        for operation in (
            PowerBaseBaseVJP,
            PowerMixedVJP,
            PowerExponentExponentVJP,
        ):
            with self.subTest(operation=operation.name):
                self.assertTrue(issubclass(operation, Operation))
                self.assertIn("forward", vars(operation))

    def test_the_mixed_partial_is_configured_by_the_operand_it_serves(self):
        operation = PowerMixedVJP(dtype=ts.float32)
        self.assertEqual(sorted(type(operation).__slots__), ["dtype"])
        self.assertIs(operation.dtype, ts.float32)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            operation.dtype = ts.float64


class TheOrderTheRulesStopAt(PowerSecondDerivativeTestCase):
    def test_a_third_derivative_reports_the_order_it_stops_at(self):
        """The third partials are not implemented, and it says so."""

        def body(backend):
            base = ts.Variable(ts.Tensor([2.0], dtype=ts.float64), requires_grad=True)
            exponent = ts.Variable(
                ts.Tensor([3.0], dtype=ts.float64), requires_grad=True
            )
            (first,) = ts.grad(ts.sum(base**exponent), (base,), create_graph=True)
            (second,) = ts.grad(ts.sum(first), (base,), create_graph=True)
            with self.assertRaisesRegex(
                NotImplementedError, "differentiated a third time"
            ):
                ts.grad(ts.sum(second), (base,))

        self.for_each_backend(body)


if __name__ == "__main__":
    unittest.main()
