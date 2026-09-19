"""D7 conformance: the derivatives of ``**`` (section 12.7).

Every expectation comes from the region table of section 12.7.2, written out
below, and from the rules G1 to G6 of section 12.7.1. No backend's gradient is
used as a semantic reference, and finite differences are used nowhere: section
12.7.4 rules them out for the NaN and convention rows, and they are not needed
for the rest.

The three classes of the table are kept distinct. ``exists`` is a derivative;
``nan`` records that no derivative exists; ``convention`` is an approved
representation of a one-sided infinite slope and is approved for exactly one
region. A zero gradient, a NaN gradient and an absent gradient are three
different things and the tests below never conflate them.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import tensors as ts

from ._support import BACKENDS, DTYPE, ArithmeticTestCase

INF = math.inf
NAN = math.nan
FLOATS = ("float32", "float64")

#: Above the CUDA fusion threshold.
FUSED_SIZE = 16_384

#: Section 12.7.2, one row per line:
#: (label, base, exponent, f, df/dx, class, df/dy, class)
REGIONS = (
    ("x > 0", 2.0, 3.0, 8.0, 12.0, "exists", 8.0 * math.log(2.0), "exists"),
    ("x < 0, y odd integral", -2.0, 3.0, -8.0, 12.0, "exists", NAN, "nan"),
    ("x < 0, y even integral", -2.0, 2.0, 4.0, -4.0, "exists", NAN, "nan"),
    ("x < 0, y non-integral", -2.0, 0.5, NAN, NAN, "nan", NAN, "nan"),
    ("x = 0, y > 1", 0.0, 2.0, 0.0, 0.0, "exists", 0.0, "exists"),
    ("x = 0, y = 1", 0.0, 1.0, 0.0, 1.0, "exists", 0.0, "exists"),
    ("x = 0, 0 < y < 1", 0.0, 0.5, 0.0, INF, "convention", 0.0, "exists"),
    ("x = 0, y = 0", 0.0, 0.0, 1.0, 0.0, "exists", NAN, "nan"),
    ("x = 0, y < 0", 0.0, -1.0, INF, NAN, "nan", NAN, "nan"),
    # Negative zero takes the x = 0 rows: it compares equal to zero, and the
    # table states +inf for 0 < y < 1 with no sign variant approved.
    ("x = -0.0, y = 1", -0.0, 1.0, -0.0, 1.0, "exists", 0.0, "exists"),
    ("x = -0.0, 0 < y < 1", -0.0, 0.5, 0.0, INF, "convention", 0.0, "exists"),
    ("x = -0.0, y > 1", -0.0, 2.0, 0.0, 0.0, "exists", 0.0, "exists"),
)


class GradientTestCase(ArithmeticTestCase):
    """Comparison that keeps the three classes of section 12.7.2 apart."""

    def assertClassified(self, produced, expected, context=""):
        """Exact for zero and infinity, by classification for NaN."""
        if expected != expected:
            self.assertTrue(
                produced != produced, f"{context}: expected NaN, got {produced!r}"
            )
            return
        if math.isinf(expected):
            self.assertTrue(
                math.isinf(produced) and (produced > 0) == (expected > 0),
                f"{context}: expected {expected!r}, got {produced!r}",
            )
            return
        if expected == 0.0:
            self.assertEqual(produced, 0.0, f"{context}: got {produced!r}")
            return
        self.assertTrue(
            math.isclose(produced, expected, rel_tol=2.0**-20, abs_tol=0.0),
            f"{context}: expected {expected!r}, got {produced!r}",
        )

    def variables(self, base, exponent, dtype_name, *, count=1):
        return (
            ts.Variable(
                ts.Tensor([base] * count, dtype=DTYPE[dtype_name]), requires_grad=True
            ),
            ts.Variable(
                ts.Tensor([exponent] * count, dtype=DTYPE[dtype_name]),
                requires_grad=True,
            ),
        )


class TheRegionTable(GradientTestCase):
    """Section 12.7.2, every row, both dtypes, every backend."""

    def test_the_forward_value_is_unaffected_by_the_derivatives(self):
        for label, x, y, f, *_ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name)
                            produced = (base**exponent).data.tolist()[0]
                        self.assertClassified(produced, f, label)

    def test_both_derivatives_requested_together(self):
        for label, x, y, _, dx, _, dy, _ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name)
                            got_base, got_exponent = ts.grad(
                                base**exponent, [base, exponent]
                            )
                            values = (got_base.tolist()[0], got_exponent.tolist()[0])
                        self.assertClassified(values[0], dx, f"{label} d/dx")
                        self.assertClassified(values[1], dy, f"{label} d/dy")

    def test_the_base_derivative_requested_alone(self):
        for label, x, y, _, dx, _, _, _ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name)
                            (produced,) = ts.grad(base**exponent, [base])
                        self.assertClassified(produced.tolist()[0], dx, f"{label} d/dx")

    def test_the_exponent_derivative_requested_alone(self):
        for label, x, y, _, _, _, dy, _ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name)
                            (produced,) = ts.grad(base**exponent, [exponent])
                        self.assertClassified(produced.tolist()[0], dy, f"{label} d/dy")

    def test_no_region_raises(self):
        """Rule G2: an absent derivative is NaN, never an exception."""
        for label, x, y, *_ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name)
                            ts.grad(base**exponent, [base, exponent])
                            base, exponent = self.variables(x, y, dtype_name)
                            ts.backward(base**exponent)

    def test_gradient_dtype_and_shape(self):
        """Rule G5, and the shape each gradient is reduced to."""
        for label, x, y, *_ in REGIONS:
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(region=label, dtype=dtype_name, backend=backend):
                        with ts.use_backend(backend):
                            base, exponent = self.variables(x, y, dtype_name, count=5)
                            got_base, got_exponent = ts.grad(
                                base**exponent, [base, exponent]
                            )
                            for gradient in (got_base, got_exponent):
                                self.assertIs(gradient.dtype, DTYPE[dtype_name])
                                self.assertEqual(gradient.shape, (5,))

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_gradients_stay_in_cuda_storage(self):
        for label, x, y, *_ in REGIONS:
            for dtype_name in FLOATS:
                with self.subTest(region=label, dtype=dtype_name):
                    with ts.use_backend("cuda"):
                        base, exponent = self.variables(x, y, dtype_name, count=64)
                        got_base, got_exponent = ts.grad(
                            base**exponent, [base, exponent]
                        )
                        for gradient in (got_base, got_exponent):
                            self.assertEqual(
                                type(gradient._storage).__name__, "CudaStorage", label
                            )


class TheWorkedExample(GradientTestCase):
    """Section 12.7.3, asserted on its own because it is the reported case."""

    def test_a_valid_base_gradient_survives_an_undefined_exponent_gradient(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable([-2.0], requires_grad=True)
                exponent = ts.Variable([3.0], requires_grad=True)
                output = base**exponent

                self.assertEqual(output.data.tolist(), [-8.0])
                got_base, got_exponent = ts.grad(output, [base, exponent])
                self.assertEqual(got_base.tolist(), [12.0])
                self.assertTrue(math.isnan(got_exponent.tolist()[0]))

    def test_the_same_through_backward(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable([-2.0], requires_grad=True)
                exponent = ts.Variable([3.0], requires_grad=True)
                ts.backward(base**exponent)
                self.assertEqual(base.grad.tolist(), [12.0])
                self.assertTrue(math.isnan(exponent.grad.tolist()[0]))


class GradientsAreIndependent(GradientTestCase):
    """Rule G1, stated as the three things it requires."""

    #: Regions where one derivative exists and the other does not.
    ASYMMETRIC = (
        ("x < 0, y integral", -2.0, 3.0, 12.0, NAN),
        ("x = 0, y = 0", 0.0, 0.0, 0.0, NAN),
        ("x = 0, 0 < y < 1", 0.0, 0.5, INF, 0.0),
    )

    def test_an_undefined_gradient_does_not_suppress_a_defined_one(self):
        for label, x, y, dx, dy in self.ASYMMETRIC:
            for backend in BACKENDS:
                with self.subTest(region=label, backend=backend):
                    with ts.use_backend(backend):
                        base, exponent = self.variables(x, y, "float64")
                        alone = ts.grad(base**exponent, [base])[0].tolist()[0]
                        base, exponent = self.variables(x, y, "float64")
                        together = ts.grad(base**exponent, [base, exponent])
                    self.assertClassified(alone, dx, f"{label} alone")
                    self.assertClassified(together[0].tolist()[0], dx, f"{label} both")
                    self.assertClassified(together[1].tolist()[0], dy, f"{label} both")

    def test_an_unrequested_gradient_is_none_not_zero_and_not_nan(self):
        """The three outcomes are distinct and must stay distinguishable."""
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable([-2.0], requires_grad=True)
                exponent = ts.Variable([3.0], requires_grad=False)
                ts.backward(base**exponent)
                self.assertEqual(base.grad.tolist(), [12.0])
                self.assertIsNone(
                    exponent.grad, "an unrequested gradient must not be computed"
                )

    def test_an_unrequested_gradient_is_not_computed(self):
        """Rule G1: only a requested VJP runs."""
        import importlib

        module = importlib.import_module("tensors.operations.arithmetic.power")
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                calls = []
                original = module.execute_power_exponent_gradient

                def spy(*arguments, **keywords):
                    calls.append(1)
                    return original(*arguments, **keywords)

                with patch.object(module, "execute_power_exponent_gradient", spy):
                    with ts.use_backend(backend):
                        base, exponent = self.variables(-2.0, 3.0, "float64")
                        ts.grad(base**exponent, [base])
                self.assertEqual(
                    calls, [], "the exponent VJP ran although it was not requested"
                )


class OperandDtypesAndShapes(GradientTestCase):
    """Rule G5, and the reduction of a broadcast VJP."""

    def test_each_gradient_takes_its_own_operand_dtype(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(
                    ts.Tensor([2.0], dtype=ts.float32), requires_grad=True
                )
                exponent = ts.Variable(
                    ts.Tensor([3.0], dtype=ts.float64), requires_grad=True
                )
                output = base**exponent
                self.assertIs(output.dtype, ts.float64)

                got_base, got_exponent = ts.grad(output, [base, exponent])
                # Not the upstream gradient's dtype, which is float64 for both.
                self.assertIs(got_base.dtype, ts.float32)
                self.assertIs(got_exponent.dtype, ts.float64)

    def test_the_reverse_dtype_pairing(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(
                    ts.Tensor([2.0], dtype=ts.float64), requires_grad=True
                )
                exponent = ts.Variable(
                    ts.Tensor([3.0], dtype=ts.float32), requires_grad=True
                )
                got_base, got_exponent = ts.grad(base**exponent, [base, exponent])
                self.assertIs(got_base.dtype, ts.float64)
                self.assertIs(got_exponent.dtype, ts.float32)

    def test_a_broadcast_vjp_is_reduced_to_each_operand(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(
                    ts.Tensor([[2.0], [3.0], [4.0]], dtype=ts.float64),
                    requires_grad=True,
                )
                exponent = ts.Variable(
                    ts.Tensor([[2.0, 3.0]], dtype=ts.float64), requires_grad=True
                )
                output = base**exponent
                self.assertEqual(output.shape, (3, 2))

                got_base, got_exponent = ts.grad(output, [base, exponent])
                self.assertEqual(got_base.shape, (3, 1))
                self.assertEqual(got_exponent.shape, (1, 2))

                # d/dx summed over the exponent axis: 2x + 3x**2.
                self.assertEqual(
                    got_base.tolist(),
                    [2 * 2 + 3 * 4, 2 * 3 + 3 * 9, 2 * 4 + 3 * 16],
                )

    def test_a_broadcast_vjp_with_mixed_dtypes(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(
                    ts.Tensor([[2.0], [3.0]], dtype=ts.float32), requires_grad=True
                )
                exponent = ts.Variable(
                    ts.Tensor([[2.0, 3.0]], dtype=ts.float64), requires_grad=True
                )
                got_base, got_exponent = ts.grad(base**exponent, [base, exponent])
                self.assertIs(got_base.dtype, ts.float32)
                self.assertIs(got_exponent.dtype, ts.float64)
                self.assertEqual(got_base.shape, (2, 1))
                self.assertEqual(got_exponent.shape, (1, 2))


class EagerGraphAndFusedAgree(GradientTestCase):
    """The same region classifications through every execution path."""

    def test_graph_replay_matches_eager(self):
        from tensors.graph import Computation

        for label, x, y, _, dx, _, dy, _ in REGIONS:
            for backend in BACKENDS:
                with self.subTest(region=label, backend=backend):
                    with ts.use_backend(backend):
                        base, exponent = self.variables(x, y, "float64")
                        program = Computation(base**exponent)
                        program.forward()
                        got_base, got_exponent = ts.grad(
                            base**exponent, [base, exponent]
                        )
                    self.assertClassified(got_base.tolist()[0], dx, f"{label} d/dx")
                    self.assertClassified(got_exponent.tolist()[0], dy, f"{label} d/dy")

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_fused_backward_matches_the_table(self):
        """A second step forces fusion; the fused kernel is asserted reached."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading

        for label, x, y, _, dx, _, dy, _ in REGIONS:
            for dtype_name in FLOATS:
                with self.subTest(region=label, dtype=dtype_name):
                    with ts.use_backend("cuda"):
                        base, exponent = self.variables(
                            x, y, dtype_name, count=FUSED_SIZE
                        )
                        with patch.object(
                            cuda_backend,
                            "fused_elementwise_backward",
                            wraps=cuda_backend.fused_elementwise_backward,
                        ) as fused:
                            loading._clear_backend_kernel_cache()
                            got_base, got_exponent = ts.grad(
                                (base**exponent) * 1.0, [base, exponent]
                            )
                            reached = fused.called
                        values = (got_base.tolist()[0], got_exponent.tolist()[0])
                        shapes = (got_base.shape, got_exponent.shape)
                    self.assertTrue(reached, f"{label}: fused backward not reached")
                    self.assertClassified(values[0], dx, f"{label} fused d/dx")
                    self.assertClassified(values[1], dy, f"{label} fused d/dy")
                    self.assertEqual(shapes, ((FUSED_SIZE,), (FUSED_SIZE,)))

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_a_fused_scalar_exponent_matches_the_table(self):
        """The generated kernel specialises a literal exponent; check it."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading

        cases = (
            (2.0, 3.0, 12.0),
            (0.0, 2.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.5, INF),
            (0.0, 0.0, 0.0),
            (0.0, -1.0, NAN),
            (-2.0, 0.5, NAN),
            (-2.0, 3.0, 12.0),
        )
        for x, y, dx in cases:
            with self.subTest(base=x, exponent=y):
                with ts.use_backend("cuda"):
                    base = ts.Variable(
                        ts.Tensor([x] * FUSED_SIZE, dtype=ts.float64),
                        requires_grad=True,
                    )
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise_backward",
                        wraps=cuda_backend.fused_elementwise_backward,
                    ) as fused:
                        loading._clear_backend_kernel_cache()
                        (produced,) = ts.grad((base**y) * 1.0, [base])
                        reached = fused.called
                    value = produced.tolist()[0]
                self.assertTrue(reached, "the fused backward kernel was not reached")
                self.assertClassified(value, dx, f"{x!r} ** {y!r}")


@unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
class CudaExecutesAndStaysResident(GradientTestCase):
    """Rules G3 and G6 on the device."""

    def _counting_device_reads(self):
        import contextlib

        import tensors.tensor as tensor_module

        class Counter:
            count = 0

        @contextlib.contextmanager
        def counting():
            counter = Counter()
            original = tensor_module.Tensor._data.fget

            def counted(self):
                counter.count += 1
                return original(self)

            tensor_module.Tensor._data = property(counted)
            try:
                yield counter
            finally:
                tensor_module.Tensor._data = property(original)

        return counting()

    def _reference_calls(self):
        """A context that counts Python-reference gradient kernel calls."""
        import contextlib
        import importlib

        base_module = importlib.import_module(
            "tensors.backend.python.kernels.elementwise.power_base_gradient"
        )
        exponent_module = importlib.import_module(
            "tensors.backend.python.kernels.elementwise.power_exponent_gradient"
        )

        @contextlib.contextmanager
        def counting():
            calls = []
            originals = (
                base_module.power_base_gradient,
                exponent_module.power_exponent_gradient,
            )

            def spy_base(*arguments, **keywords):
                calls.append("base")
                return originals[0](*arguments, **keywords)

            def spy_exponent(*arguments, **keywords):
                calls.append("exponent")
                return originals[1](*arguments, **keywords)

            with (
                patch.object(base_module, "power_base_gradient", spy_base),
                patch.object(exponent_module, "power_exponent_gradient", spy_exponent),
            ):
                yield calls

        return counting()

    def test_no_host_read_classifies_a_region(self):
        """Rule G3: nothing is materialised to decide a derivative's region."""
        sizes = (1, 64, 100_000)
        mixed = [-2.0, 0.0, 2.0, INF, NAN, -0.0]
        for size in sizes:
            with self.subTest(size=size):
                with ts.use_backend("cuda"):
                    # Every region in one tensor, so the kernel must classify
                    # elementwise rather than branch on a summary it read.
                    values = (mixed * (size // len(mixed) + 1))[:size]
                    base = ts.Variable(
                        ts.Tensor(values, dtype=ts.float64) + 0.0, requires_grad=True
                    )
                    exponent = ts.Variable(
                        ts.Tensor(
                            [0.5, 1.0, 2.0, -1.0, 0.0, 3.0][: len(values)]
                            * (size // 6 + 1),
                            dtype=ts.float64,
                        )[:size]
                        + 0.0,
                        requires_grad=True,
                    )
                    output = base**exponent
                    with self._counting_device_reads() as reads:
                        got_base, got_exponent = ts.grad(output, [base, exponent])
                        self.assertEqual(
                            type(got_base._storage).__name__, "CudaStorage"
                        )
                        self.assertEqual(
                            type(got_exponent._storage).__name__, "CudaStorage"
                        )
                self.assertEqual(
                    reads.count, 0, f"size {size} materialised a tensor on the host"
                )

    def test_no_value_dependent_fallback(self):
        """Rule G6: every size and every operand value stays on CUDA."""
        cases = (
            ("ordinary", 2.0, 3.0),
            ("negative base", -2.0, 3.0),
            ("zero base", 0.0, 0.5),
            ("infinite base", INF, 2.0),
            ("nan base", NAN, 2.0),
            ("subnormal base", 5e-324, 2.0),
        )
        for label, x, y in cases:
            for size in (1, 4, 64, 100_000):
                with self.subTest(case=label, size=size):
                    with self._reference_calls() as calls:
                        with ts.use_backend("cuda"):
                            base, exponent = self.variables(x, y, "float64", count=size)
                            got_base, got_exponent = ts.grad(
                                base**exponent, [base, exponent]
                            )
                            residency = (
                                type(got_base._storage).__name__,
                                type(got_exponent._storage).__name__,
                            )
                    self.assertEqual(
                        calls, [], f"{label} at size {size} fell back to Python"
                    )
                    self.assertEqual(residency, ("CudaStorage", "CudaStorage"))

    def test_both_gradients_are_returned_when_only_one_has_nans(self):
        """A mixed tensor: valid and undefined regions side by side."""
        pattern = [2.0, -2.0, 0.0, 3.0]
        size = 4096
        with ts.use_backend("cuda"):
            values = (pattern * (size // len(pattern)))[:size]
            base = ts.Variable(ts.Tensor(values, dtype=ts.float64), requires_grad=True)
            exponent = ts.Variable(
                ts.Tensor([3.0] * size, dtype=ts.float64), requires_grad=True
            )
            got_base, got_exponent = ts.grad(base**exponent, [base, exponent])
            base_values = got_base.tolist()[:4]
            exponent_values = got_exponent.tolist()[:4]

        # d/dx = 3x**2 exists at every one of these bases.
        self.assertEqual(base_values, [12.0, 12.0, 0.0, 27.0])
        # d/dy exists only where the base is positive.
        self.assertAlmostEqual(exponent_values[0], 8.0 * math.log(2.0))
        self.assertTrue(math.isnan(exponent_values[1]))
        self.assertEqual(exponent_values[2], 0.0)
        self.assertAlmostEqual(exponent_values[3], 27.0 * math.log(3.0))

    def test_a_subnormal_gradient_is_not_flushed(self):
        """Section 5.4 in the gradient path, on binary32.

        ``d/dy`` of ``1e-10 ** 4`` is a binary32 subnormal. Narrowing the
        binary64 result with ``astype`` flushed it to ``-0.0``; the PTX
        conversion keeps it.
        """
        with ts.use_backend("cuda"):
            base = ts.Variable(
                ts.Tensor([1e-10] * 64, dtype=ts.float32), requires_grad=True
            )
            exponent = ts.Variable(
                ts.Tensor([4.0] * 64, dtype=ts.float32), requires_grad=True
            )
            (produced,) = ts.grad(base**exponent, [exponent])
            value = produced.tolist()[0]
        self.assertNotEqual(value, 0.0, "the subnormal gradient was flushed")
        self.assertLess(abs(value), 1.1754943508222875e-38, "it should be subnormal")
        self.assertLess(value, 0.0)


class ANanExponentPropagates(GradientTestCase):
    """Section 12.7.2: a NaN operand propagates to a NaN gradient.

    The fused CUDA base gradient classified a zero base by comparing its
    exponent against zero and one. Every ordered comparison is false for NaN,
    so a NaN exponent fell through the whole chain to the last branch, which
    assumes ``0 < y < 1`` and returns the ``+inf`` convention. The convention
    is approved for that region and for no other, least of all for an operand
    that is not a number at all.

    Eager and unfused execution were already correct, so a test that only
    reads the result proves nothing: it has to establish that the *fused*
    power VJP produced the value.
    """

    BASE = 0.0
    EXPONENT = NAN

    def _power_gradient_calls(self):
        """Count unfused power-gradient kernel calls, to exclude a fallback."""
        import contextlib
        import importlib

        module = importlib.import_module("tensors.operations.arithmetic.power")

        @contextlib.contextmanager
        def counting():
            calls = []
            original = module.execute_power_base_gradient

            def spy(*arguments, **keywords):
                calls.append(1)
                return original(*arguments, **keywords)

            with patch.object(module, "execute_power_base_gradient", spy):
                yield calls

        return counting()

    def test_eager_gives_a_nan_base_gradient(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        base = ts.Variable(
                            ts.Tensor([self.BASE], dtype=DTYPE[dtype_name]),
                            requires_grad=True,
                        )
                        exponent = ts.Variable(
                            ts.Tensor([self.EXPONENT], dtype=DTYPE[dtype_name]),
                            requires_grad=False,
                        )
                        output = base**exponent
                        forward = output.data.tolist()[0]
                        (produced,) = ts.grad(output, [base])
                        value = produced.tolist()[0]
                        self.assertIs(produced.dtype, DTYPE[dtype_name])
                        self.assertEqual(produced.shape, (1,))
                    self.assertTrue(math.isnan(forward), "forward must be NaN")
                    self.assertTrue(math.isnan(value), f"got {value!r}")

    def test_unfused_graph_replay_gives_a_nan_base_gradient(self):
        from tensors.graph import Computation

        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        base = ts.Variable(
                            ts.Tensor([self.BASE], dtype=DTYPE[dtype_name]),
                            requires_grad=True,
                        )
                        exponent = ts.Variable(
                            ts.Tensor([self.EXPONENT], dtype=DTYPE[dtype_name]),
                            requires_grad=False,
                        )
                        forward = Computation(base**exponent).forward().tolist()[0]
                        (produced,) = ts.grad(base**exponent, [base])
                        value = produced.tolist()[0]
                    self.assertTrue(math.isnan(forward), "forward must be NaN")
                    self.assertTrue(math.isnan(value), f"got {value!r}")

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_fused_cuda_backward_gives_a_nan_base_gradient(self):
        """The case the correction is for, with the fused path established."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading

        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                with self._power_gradient_calls() as unfused_calls:
                    with ts.use_backend("cuda"):
                        base = ts.Variable(
                            ts.Tensor(
                                [self.BASE] * FUSED_SIZE, dtype=DTYPE[dtype_name]
                            ),
                            requires_grad=True,
                        )
                        exponent = ts.Variable(
                            ts.Tensor(
                                [self.EXPONENT] * FUSED_SIZE,
                                dtype=DTYPE[dtype_name],
                            ),
                            requires_grad=False,
                        )
                        with patch.object(
                            cuda_backend,
                            "fused_elementwise_backward",
                            wraps=cuda_backend.fused_elementwise_backward,
                        ) as fused:
                            loading._clear_backend_kernel_cache()
                            # A second step forces fusion; only the base
                            # gradient is requested, so the chain is not sent
                            # back to ordinary execution for an external one.
                            (produced,) = ts.grad((base**exponent) * 1.0, [base])
                            reached = fused.called
                        value = produced.tolist()[0]
                        residency = type(produced._storage).__name__
                        dtype = produced.dtype
                        shape = produced.shape

                self.assertTrue(reached, "the fused backward kernel was not reached")
                self.assertEqual(
                    unfused_calls,
                    [],
                    "the power step fell back to the unfused gradient kernel, "
                    "so the fused VJP was not what produced this value",
                )
                self.assertTrue(math.isnan(value), f"got {value!r}")
                self.assertIs(dtype, DTYPE[dtype_name])
                self.assertEqual(shape, (FUSED_SIZE,))
                self.assertEqual(residency, "CudaStorage")

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_the_fused_case_reads_nothing_back_to_the_host(self):
        """Rule G3 holds for the corrected classification too."""
        import contextlib

        import tensors.backend.cuda.kernels as cuda_backend
        import tensors.tensor as tensor_module
        from tensors.backend import loading

        @contextlib.contextmanager
        def counting():
            reads = []
            original = tensor_module.Tensor._data.fget

            def counted(self):
                reads.append(1)
                return original(self)

            tensor_module.Tensor._data = property(counted)
            try:
                yield reads
            finally:
                tensor_module.Tensor._data = property(original)

        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name):
                with ts.use_backend("cuda"):
                    base = ts.Variable(
                        ts.Tensor([self.BASE] * FUSED_SIZE, dtype=DTYPE[dtype_name]),
                        requires_grad=True,
                    )
                    exponent = ts.Variable(
                        ts.Tensor(
                            [self.EXPONENT] * FUSED_SIZE, dtype=DTYPE[dtype_name]
                        ),
                        requires_grad=False,
                    )
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise_backward",
                        wraps=cuda_backend.fused_elementwise_backward,
                    ) as fused:
                        loading._clear_backend_kernel_cache()
                        program = (base**exponent) * 1.0
                        with counting() as reads:
                            (produced,) = ts.grad(program, [base])
                            self.assertEqual(
                                type(produced._storage).__name__, "CudaStorage"
                            )
                        self.assertTrue(fused.called)
                self.assertEqual(reads, [], "a tensor was materialised on the host")

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_the_other_zero_base_rows_are_unchanged(self):
        """The correction must not swallow the +inf convention it sits above."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading

        rows = (
            ("y < 0", -1.0, NAN),
            ("y = 0", 0.0, 0.0),
            ("0 < y < 1", 0.5, INF),
            ("y = 1", 1.0, 1.0),
            ("y > 1", 2.0, 0.0),
            ("y is NaN", NAN, NAN),
        )
        for label, y, expected in rows:
            with self.subTest(row=label):
                with ts.use_backend("cuda"):
                    base = ts.Variable(
                        ts.Tensor([0.0] * FUSED_SIZE, dtype=ts.float64),
                        requires_grad=True,
                    )
                    exponent = ts.Variable(
                        ts.Tensor([y] * FUSED_SIZE, dtype=ts.float64),
                        requires_grad=False,
                    )
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise_backward",
                        wraps=cuda_backend.fused_elementwise_backward,
                    ) as fused:
                        loading._clear_backend_kernel_cache()
                        (produced,) = ts.grad((base**exponent) * 1.0, [base])
                        reached = fused.called
                    value = produced.tolist()[0]
                self.assertTrue(reached, f"{label}: fused backward not reached")
                self.assertClassified(value, expected, label)

    def test_the_literal_exponent_generator_cannot_misclassify_nan(self):
        """That path is unreachable publicly; the guard is there regardless.

        The fusion planner writes ``None`` into every step's scalar field, at
        each of its construction sites, and it is the only producer of fused
        steps — so a fused power step always carries its exponent as a tensor
        operand and this generator never sees a literal. It is still made
        NaN-safe, because its comparisons are all false for NaN and would
        otherwise select the ``0 < y < 1`` convention.
        """
        from tensors.backend.cuda.kernels.fusion.expressions import (
            _NAN,
            _zero_base_base_derivative,
        )

        self.assertEqual(_zero_base_base_derivative(NAN, "g"), f"(g) * {_NAN}")
        # The rows it does serve are untouched.
        self.assertEqual(_zero_base_base_derivative(-1.0, "g"), f"(g) * {_NAN}")
        self.assertEqual(_zero_base_base_derivative(2.0, "g"), "(g) * 0.0")
        self.assertIn("7ff0000000000000", _zero_base_base_derivative(0.5, "g"))

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_no_fused_step_carries_a_literal_scalar(self):
        """The evidence that the literal generator is unreachable."""
        import tensors.backend as backend_module
        from tensors.backend import loading

        observed = []
        original_forward = backend_module.execute_fused_elementwise
        original_backward = backend_module.execute_fused_elementwise_backward

        def forward_spy(values, steps, **keywords):
            observed.extend(steps)
            return original_forward(values, steps, **keywords)

        def backward_spy(values, upstream, steps, **keywords):
            observed.extend(steps)
            return original_backward(values, upstream, steps, **keywords)

        expressions = (
            lambda v, e: (v**0.5) * 1.0,
            lambda v, e: (v**NAN) * 1.0,
            lambda v, e: (v**e) * 1.0,
            lambda v, e: (v + 2.0) ** 0.5,
            lambda v, e: v * 3.0 + 1.0,
            lambda v, e: (2.0**v) * 1.0,
        )
        with (
            patch.object(backend_module, "execute_fused_elementwise", forward_spy),
            patch.object(
                backend_module, "execute_fused_elementwise_backward", backward_spy
            ),
        ):
            for build in expressions:
                with ts.use_backend("cuda"):
                    variable = ts.Variable(
                        ts.Tensor([2.0] * FUSED_SIZE, dtype=ts.float64),
                        requires_grad=True,
                    )
                    other = ts.Variable(
                        ts.Tensor([3.0] * FUSED_SIZE, dtype=ts.float64),
                        requires_grad=False,
                    )
                    loading._clear_backend_kernel_cache()
                    ts.grad(build(variable, other), [variable])

        self.assertTrue(observed, "no fused step was observed")
        self.assertEqual(
            [step for step in observed if step[1] is not None],
            [],
            "a fused step carried a literal scalar; the literal-exponent "
            "generator is reachable after all and needs its own coverage",
        )


class HigherOrderDifferentiation(GradientTestCase):
    """Section 12.7 does not weaken the existing contract on smooth inputs."""

    def test_create_graph_on_ordinary_inputs(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable([2.0], requires_grad=True)
                exponent = ts.Variable([3.0], requires_grad=True)
                first = ts.grad(base**exponent, base, create_graph=True)
                self.assertEqual(first.data.tolist(), [12.0])

                # d2/dx2 of x**y is y(y-1)x**(y-2) = 3 * 2 * 2 = 12.
                second = ts.grad(first, base)
                self.assertEqual(second.tolist(), [12.0])

    def test_a_mixed_second_derivative(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable([2.0], requires_grad=True)
                exponent = ts.Variable([3.0], requires_grad=True)
                first = ts.grad(base**exponent, base, create_graph=True)
                mixed = ts.grad(first, exponent)
                # d/dy of y x**(y-1) = x**(y-1) (1 + y ln x) = 4(1 + 3 ln 2).
                self.assertAlmostEqual(
                    mixed.tolist()[0], 4.0 * (1.0 + 3.0 * math.log(2.0))
                )


class OrdinaryGradientAccuracy(GradientTestCase):
    """Section 12.7.4, on the rows where a derivative exists and is finite.

    The reference is :mod:`_gradient_reference`, which resolves an enclosure
    of the derivative and refuses to answer otherwise. Nothing here is
    compared against another backend, and no finite difference is taken.

    **The bound below is not section 12.6.2's.** Section 12.7.4 is explicit
    that the 2 and 4 ULP figures apply to ``**`` itself and are *not* imposed
    on a complete gradient expression, which composes a power, a logarithm and
    two multiplications. The 64 ULP used here is a detection threshold: it is
    wide enough that the accumulation of a correct implementation passes, and
    narrow enough that a wrong formula or a lost intermediate does not. The
    maximum actually observed is reported in the assertion message, and is far
    below it. It certifies nothing about accuracy and must not be read as a
    guarantee.
    """

    #: Wide enough for an accumulation, narrow enough to catch a wrong formula.
    DETECTION_THRESHOLD = 64

    BASES = (0.5, 0.75, 1.5, 2.0, 3.0, 7.0, 97.0, 1234.5, 1e10, 1e-10)
    EXPONENTS = (0.5, 1.5, 2.5, 3.0, -0.5, -1.5, 0.3333333333333333, 4.0)
    #: Negative bases with integral exponents, where only d/dx exists.
    NEGATIVE = ((-2.0, 3.0), (-2.0, 4.0), (-1.5, 5.0), (-7.5, -3.0))

    def measure(self, dtype_name, backend, pairs):
        from . import _accuracy
        from . import _gradient_reference as reference

        worst_base = 0
        worst_exponent = 0
        failures = []
        for x, y in pairs:
            with ts.use_backend(backend):
                base, exponent = self.variables(x, y, dtype_name)
                stored_base = base.data.tolist()[0]
                stored_exponent = exponent.data.tolist()[0]
                got_base, got_exponent = ts.grad(base**exponent, [base, exponent])
                values = (got_base.tolist()[0], got_exponent.tolist()[0])

            expected = reference.base_derivative(
                stored_base, stored_exponent, dtype_name
            )
            comparison = _accuracy.compare(values[0], expected, dtype_name)
            if comparison.distance is not None:
                worst_base = max(worst_base, comparison.distance)
                if comparison.distance > self.DETECTION_THRESHOLD:
                    failures.append(
                        f"d/dx {stored_base!r} ** {stored_exponent!r}: "
                        f"{comparison.distance} ULP, got {values[0]!r}, "
                        f"expected {expected!r}"
                    )
            elif not comparison.conforms:
                failures.append(
                    f"d/dx {stored_base!r} ** {stored_exponent!r}: "
                    f"{comparison.classification}, got {values[0]!r}, "
                    f"expected {expected!r}"
                )

            if stored_base <= 0.0:
                continue
            expected = reference.exponent_derivative(
                stored_base, stored_exponent, dtype_name
            )
            comparison = _accuracy.compare(values[1], expected, dtype_name)
            if comparison.distance is not None:
                worst_exponent = max(worst_exponent, comparison.distance)
                if comparison.distance > self.DETECTION_THRESHOLD:
                    failures.append(
                        f"d/dy {stored_base!r} ** {stored_exponent!r}: "
                        f"{comparison.distance} ULP, got {values[1]!r}, "
                        f"expected {expected!r}"
                    )
            elif not comparison.conforms:
                failures.append(
                    f"d/dy {stored_base!r} ** {stored_exponent!r}: "
                    f"{comparison.classification}, got {values[1]!r}, "
                    f"expected {expected!r}"
                )
        return worst_base, worst_exponent, failures

    def test_ordinary_gradients_match_the_high_precision_reference(self):
        pairs = [(x, y) for x in self.BASES for y in self.EXPONENTS]
        pairs += list(self.NEGATIVE)
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    worst_base, worst_exponent, failures = self.measure(
                        dtype_name, backend, pairs
                    )
                    self.assertFalse(
                        failures,
                        f"{backend}/{dtype_name}: {len(failures)} of {len(pairs)} "
                        f"beyond {self.DETECTION_THRESHOLD} ULP "
                        f"(worst d/dx {worst_base}, worst d/dy {worst_exponent}):\n"
                        + "\n".join(failures[:6]),
                    )

    def test_the_reference_agrees_with_exactly_known_derivatives(self):
        """Validation of the reference itself, before it judges anything."""
        from . import _gradient_reference as reference

        for x, y, expected in (
            (2.0, 3.0, 12.0),
            (3.0, 2.0, 6.0),
            (-2.0, 3.0, 12.0),
            (-2.0, 2.0, -4.0),
            (0.5, 4.0, 0.5),
            (4.0, 0.5, 0.25),
            (2.0, -1.0, -0.25),
        ):
            with self.subTest(base=x, exponent=y):
                self.assertEqual(reference.base_derivative(x, y, "float64"), expected)

        for x, y, expected in (
            (2.0, 3.0, 8.0 * math.log(2.0)),
            (3.0, 2.0, 9.0 * math.log(3.0)),
            (2.0, 0.0, math.log(2.0)),
        ):
            with self.subTest(base=x, exponent=y):
                self.assertAlmostEqual(
                    reference.exponent_derivative(x, y, "float64"), expected, places=14
                )

    def test_the_reference_refuses_the_classified_rows(self):
        """It answers only where a derivative exists; the rest are literals."""
        from . import _gradient_reference as reference

        for x, y in ((0.0, 2.0), (-2.0, 0.5), (INF, 2.0), (NAN, 2.0)):
            with self.subTest(base=x, exponent=y):
                with self.assertRaises(ValueError):
                    reference.base_derivative(x, y, "float64")
        for x, y in ((0.0, 2.0), (-2.0, 3.0), (INF, 2.0)):
            with self.subTest(base=x, exponent=y):
                with self.assertRaises(ValueError):
                    reference.exponent_derivative(x, y, "float64")


class IntegerTensorsRemainUndifferentiable(ArithmeticTestCase):
    """Rule G4, which section 12.4 never interacts with."""

    def test_an_integer_variable_cannot_require_gradients(self):
        for dtype_name in ("int32", "int64"):
            with self.subTest(dtype=dtype_name):
                with self.assertRaises(ValueError):
                    ts.Variable(
                        ts.Tensor([2], dtype=DTYPE[dtype_name]), requires_grad=True
                    )

    def test_integer_exponentiation_is_unchanged(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                produced = ts.Tensor([2, 3], dtype=ts.int32) ** ts.Tensor(
                    [10, 4], dtype=ts.int32
                )
                self.assertEqual(produced.tolist(), [1024, 81])
                with self.assertRaises(ValueError):
                    ts.Tensor([2], dtype=ts.int32) ** ts.Tensor([-1], dtype=ts.int32)


if __name__ == "__main__":
    unittest.main()
