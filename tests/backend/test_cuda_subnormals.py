"""CUDA binary32 subnormal handling for the unary elementwise kernels.

Audit finding **D-1** (`docs/numerical-completeness-audit.md`): the eager CUDA
binary32 kernels flushed subnormal operands to zero. The cause was the format
crossing, not the mathematics — the kernels evaluate in a binary64 working
precision, and ``astype`` flushes a binary32 subnormal *on the way up*, so the
operand was already zero before any function was applied.

Expectations here are established independently of any backend:

* ``abs`` and ``sign`` are **exact** (class E in the audit). Their results
  follow from the operand's bit pattern, so they are derived from it.
* ``sqrt`` is **correctly rounded** (class R). The expected value is computed
  by exact integer arithmetic in :func:`correctly_rounded_sqrt`, which
  consults nothing but the operand.
* For the odd functions, every one of ``sin``, ``tan``, ``arcsin``,
  ``arctan``, ``arcsinh``, ``arctanh``, ``sinh`` and ``tanh`` has the Taylor
  expansion ``f(x) = x + O(x³)``. At binary32 subnormal magnitudes ``|x| ≤
  2**-126``, so ``|x³| ≤ 2**-378``, which is below half an ulp of ``x`` by
  more than two hundred binary orders of magnitude. The correctly rounded
  result is therefore exactly ``x``, and that is a mathematical fact rather
  than an observation of what a backend returns.
* ``relu`` is exact by its definition.

**No accuracy tolerance is asserted anywhere in this module.** The accuracy of
these functions at ordinary magnitudes is unspecified — audit finding S-1 —
and inventing a bound here would pre-empt that decision. What is tested is the
demonstrated failure and the properties that can be established without one.
"""

from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction
from unittest.mock import patch

import tensors as ts

from tests.backend._support import requires_cuda

#: The smallest positive binary32 subnormal, and the largest.
SMALLEST = 1.401298464324817e-45
LARGEST_SUBNORMAL = 1.1754942106924411e-38
MIN_NORMAL = 1.1754943508222875e-38

#: The twelve operations audit finding D-1 names.
AFFECTED = {
    "sqrt": ts.sqrt,
    "abs": ts.abs,
    "sign": ts.sign,
    "sin": ts.sin,
    "tan": ts.tan,
    "arcsin": ts.arcsin,
    "arctan": ts.arctan,
    "arcsinh": ts.arcsinh,
    "arctanh": ts.arctanh,
    "sinh": ts.sinh,
    "tanh": ts.tanh,
    "relu": ts.relu,
}

#: Those whose Taylor expansion is ``x + O(x³)``, so a subnormal operand is
#: returned unchanged. ``sqrt``, ``abs``, ``sign`` and ``relu`` are handled by
#: their own exact rules.
ODD_ABOUT_ZERO = (
    "sin",
    "tan",
    "arcsin",
    "arctan",
    "arcsinh",
    "arctanh",
    "sinh",
    "tanh",
)


def bits32(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits32(pattern: int) -> float:
    return struct.unpack("<f", struct.pack("<I", pattern))[0]


def correctly_rounded_sqrt(value: float) -> float:
    """The correctly rounded binary32 square root, by exact integer arithmetic.

    The candidate ``y`` is accepted when ``y**2`` is at least as close to
    ``value`` as either neighbour's, with ties broken to even. Every
    comparison is between exact rationals, so nothing here approximates and
    no library's ``sqrt`` is consulted.
    """
    if value == 0.0 or value != value or math.isinf(value):
        return value
    assert value > 0.0, "negative operands are a domain question, not tested here"

    exact = Fraction(*value.as_integer_ratio())
    # Start from any nearby binary32 value and walk to the best one; the
    # integer ordering of positive binary32 values is monotone in magnitude.
    low, high = 0, bits32(3.4028234663852886e38)
    while low < high:
        middle = (low + high) // 2
        candidate = from_bits32(middle)
        if Fraction(*candidate.as_integer_ratio()) ** 2 < exact:
            low = middle + 1
        else:
            high = middle

    upper = from_bits32(low)
    lower = from_bits32(low - 1) if low else 0.0
    # Which neighbour is nearer the true root is decided by squaring their
    # midpoint: the true root lies below it exactly when its square does.
    midpoint = (
        Fraction(*lower.as_integer_ratio()) + Fraction(*upper.as_integer_ratio())
    ) / 2
    if midpoint**2 < exact:
        return upper
    if midpoint**2 > exact:
        return lower
    return upper if low % 2 == 0 else lower


@requires_cuda
class SubnormalOperandsSurvive(unittest.TestCase):
    """The reported failure, for every operation D-1 names."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def evaluate(self, function, values, dtype=ts.float32, backend="cuda"):
        with ts.use_backend(backend):
            return function(ts.Tensor(list(values), dtype=dtype)).tolist()

    def test_no_operation_returns_zero_for_a_subnormal_operand(self):
        """The audit's reproducer: every one of the twelve returned 0.0."""
        for name, function in AFFECTED.items():
            with self.subTest(operation=name):
                produced = self.evaluate(function, [SMALLEST])[0]
                self.assertNotEqual(
                    produced, 0.0, f"{name} flushed the subnormal operand"
                )

    def test_negative_subnormal_operands(self):
        for name, function in AFFECTED.items():
            # relu(x) is 0 for x < 0 and sqrt(x) has no real value there;
            # abs(x) is positive by definition. Their negative-operand rules
            # are their own and are tested with the exact operations.
            if name in {"relu", "sqrt", "abs"}:
                continue
            with self.subTest(operation=name):
                produced = self.evaluate(function, [-SMALLEST])[0]
                self.assertNotEqual(
                    produced, 0.0, f"{name} flushed the negative subnormal"
                )
                self.assertLess(produced, 0.0, f"{name} lost the sign")

    def test_across_the_whole_subnormal_range(self):
        """Not only the smallest: every binade of the subnormal range."""
        magnitudes = [from_bits32(1 << k) for k in range(23)]
        magnitudes.append(LARGEST_SUBNORMAL)
        for name in ODD_ABOUT_ZERO:
            with self.subTest(operation=name):
                produced = self.evaluate(AFFECTED[name], magnitudes)
                for index, (got, want) in enumerate(zip(produced, magnitudes)):
                    self.assertEqual(
                        bits32(got),
                        bits32(want),
                        f"{name} at 2**{index}: expected {want!r}, got {got!r}",
                    )


@requires_cuda
class ExactOperations(unittest.TestCase):
    """``abs``, ``sign`` and ``relu`` are exact; no tolerance applies."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def operands(self):
        """Subnormals of both signs, the boundaries, and ordinary values."""
        values = []
        for magnitude in (
            SMALLEST,
            2 * SMALLEST,
            LARGEST_SUBNORMAL,
            MIN_NORMAL,
            1.5,
            3.4028234663852886e38,
        ):
            values.extend((magnitude, -magnitude))
        values.extend((0.0, -0.0))
        return values

    def test_abs_clears_the_sign_bit_and_changes_nothing_else(self):
        """Derived from the bit pattern, not from another backend."""
        values = self.operands()
        with ts.use_backend("cuda"):
            produced = ts.abs(ts.Tensor(values, dtype=ts.float32)).tolist()
        for value, got in zip(values, produced):
            with self.subTest(value=value):
                self.assertEqual(
                    bits32(got),
                    bits32(value) & 0x7FFFFFFF,
                    f"abs({value!r}) gave {got!r}",
                )

    def test_sign_is_exactly_one_for_every_nonzero_finite_operand(self):
        values = [v for v in self.operands() if v != 0.0]
        with ts.use_backend("cuda"):
            produced = ts.sign(ts.Tensor(values, dtype=ts.float32)).tolist()
        for value, got in zip(values, produced):
            with self.subTest(value=value):
                self.assertEqual(got, 1.0 if value > 0 else -1.0, f"sign({value!r})")

    def test_relu_returns_the_operand_or_zero(self):
        values = self.operands()
        with ts.use_backend("cuda"):
            produced = ts.relu(ts.Tensor(values, dtype=ts.float32)).tolist()
        for value, got in zip(values, produced):
            with self.subTest(value=value):
                if value > 0.0:
                    self.assertEqual(bits32(got), bits32(value))
                else:
                    self.assertEqual(got, 0.0)


@requires_cuda
class SquareRootIsCorrectlyRounded(unittest.TestCase):
    """Class R: the result is uniquely determined, so it is checked exactly.

    Evaluating in binary64 and rounding once is sound for a binary32 square
    root: binary64 carries 53 significand bits, and 2p + 2 = 50 bits suffice
    for the single rounding to land on the correctly rounded binary32 value.
    """

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def operands(self):
        values = [
            SMALLEST,
            2 * SMALLEST,
            3 * SMALLEST,
            LARGEST_SUBNORMAL,
            MIN_NORMAL,
            0.25,
            1.0,
            2.0,
            3.0,
            1.5,
            16.0,
            1e20,
            3.4028234663852886e38,
        ]
        values.extend(from_bits32(1 << k) for k in range(1, 23))
        return values

    def test_the_oracle_agrees_with_exactly_known_roots(self):
        """Validate the oracle before using it."""
        for value, expected in ((4.0, 2.0), (9.0, 3.0), (0.25, 0.5), (1.0, 1.0)):
            with self.subTest(value=value):
                self.assertEqual(correctly_rounded_sqrt(value), expected)

    def test_sqrt_matches_the_exact_integer_oracle(self):
        values = self.operands()
        with ts.use_backend("cuda"):
            produced = ts.sqrt(ts.Tensor(values, dtype=ts.float32)).tolist()
        for value, got in zip(values, produced):
            with self.subTest(value=value):
                self.assertEqual(
                    bits32(got),
                    bits32(correctly_rounded_sqrt(value)),
                    f"sqrt({value!r}) gave {got!r}",
                )

    def test_a_subnormal_operand_gives_a_normal_root(self):
        """The one case where a subnormal input leaves the subnormal range."""
        with ts.use_backend("cuda"):
            produced = ts.sqrt(ts.Tensor([SMALLEST], dtype=ts.float32)).tolist()[0]
        self.assertEqual(bits32(produced), bits32(correctly_rounded_sqrt(SMALLEST)))
        self.assertGreater(produced, MIN_NORMAL)


@requires_cuda
class SubnormalResultsFromNormalOperands(unittest.TestCase):
    """The result must reach the subnormal range, not stop at zero."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_an_odd_function_of_a_normal_operand_below_the_boundary(self):
        """``sin`` of the smallest normal is subnormal only by rounding down.

        The operand is normal; the mathematical result is ``x - x³/6``, which
        is below ``x`` and therefore still normal. The point of the case is
        that the kernel returns a value in that neighbourhood rather than
        zero.
        """
        for name in ODD_ABOUT_ZERO:
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    produced = AFFECTED[name](
                        ts.Tensor([MIN_NORMAL], dtype=ts.float32)
                    ).tolist()[0]
                self.assertEqual(bits32(produced), bits32(MIN_NORMAL), name)

    def test_abs_of_the_largest_subnormal(self):
        with ts.use_backend("cuda"):
            produced = ts.abs(
                ts.Tensor([-LARGEST_SUBNORMAL], dtype=ts.float32)
            ).tolist()[0]
        self.assertEqual(bits32(produced), bits32(LARGEST_SUBNORMAL))
        self.assertLess(produced, MIN_NORMAL, "the result should stay subnormal")


@requires_cuda
class EagerAndFusedAgree(unittest.TestCase):
    """A fused chain and an eager call must give the same bits (autodiff.md)."""

    FUSED_SIZE = 16_384

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_every_affected_operation_at_a_subnormal(self):
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading
        from tensors.graph import Computation

        for name, function in AFFECTED.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    operand = ts.Tensor([SMALLEST] * self.FUSED_SIZE, dtype=ts.float32)
                    eager = (function(operand) * 1.0).tolist()[0]
                    variable = ts.Variable(operand, requires_grad=False)
                    with patch.object(
                        cuda_backend,
                        "fused_elementwise",
                        wraps=cuda_backend.fused_elementwise,
                    ) as fused:
                        loading._clear_backend_kernel_cache()
                        produced = (
                            Computation(function(variable) * 1.0).forward().tolist()[0]
                        )
                        reached = fused.called
                self.assertTrue(reached, f"{name}: the fused kernel was not reached")
                self.assertEqual(
                    bits32(produced), bits32(eager), f"{name}: fused differs from eager"
                )


@requires_cuda
class ResidencyAndHostTransfers(unittest.TestCase):
    """The correction must not move work or operands to the host."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

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

    def test_results_stay_in_cuda_storage(self):
        for name, function in AFFECTED.items():
            for size in (1, 64, 100_000):
                with self.subTest(operation=name, size=size):
                    with ts.use_backend("cuda"):
                        operand = ts.Tensor([SMALLEST] * size, dtype=ts.float32)
                        produced = function(operand)
                        self.assertEqual(
                            type(produced.backend_storage).__name__, "CudaStorage"
                        )
                        self.assertIs(produced.dtype, ts.float32)

    def test_no_operand_is_read_back_to_the_host(self):
        for name, function in AFFECTED.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    operand = ts.Tensor([SMALLEST] * 4096, dtype=ts.float32) + 0.0
                    with self._counting_device_reads() as reads:
                        produced = function(operand)
                        self.assertEqual(
                            type(produced.backend_storage).__name__, "CudaStorage"
                        )
                self.assertEqual(
                    reads.count, 0, f"{name} materialised an operand on the host"
                )


@requires_cuda
class NoRegressionAtOrdinaryAndExceptionalInputs(unittest.TestCase):
    """The conversion change must not disturb anything already correct."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    ORDINARY = (0.5, 0.25, 1.0, 2.0, -0.5, -1.0, 0.75)

    def test_ordinary_operands_are_unchanged_across_backends(self):
        """Not an oracle: a divergence here would signal a regression.

        The backends were in agreement before this change at these operands,
        so a new disagreement would be evidence that the conversion altered
        an ordinary result. It is not offered as proof of correctness.
        """
        for name, function in AFFECTED.items():
            values = [v for v in self.ORDINARY if not (name == "sqrt" and v < 0)]
            if name in {"arcsin", "arctanh"}:
                values = [v for v in values if abs(v) < 1.0]
            with self.subTest(operation=name):
                results = {}
                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        results[backend] = function(
                            ts.Tensor(values, dtype=ts.float32)
                        ).tolist()
                for backend, produced in results.items():
                    for index, got in enumerate(produced):
                        self.assertEqual(
                            bits32(got),
                            bits32(results["python"][index]),
                            f"{name} on {backend} at {values[index]!r}",
                        )

    def test_exceptional_values_are_unchanged(self):
        cases = ((0.0, "zero"), (-0.0, "negative zero"), (float("nan"), "nan"))
        for name, function in AFFECTED.items():
            for value, label in cases:
                with self.subTest(operation=name, case=label):
                    produced = {}
                    for backend in ts.available_backends():
                        with ts.use_backend(backend):
                            try:
                                produced[backend] = function(
                                    ts.Tensor([value], dtype=ts.float32)
                                ).tolist()[0]
                            except Exception as error:  # a domain rule, not a bug
                                produced[backend] = type(error).__name__
                    reference = produced["python"]
                    for backend, got in produced.items():
                        if isinstance(reference, float) and reference != reference:
                            self.assertTrue(
                                isinstance(got, float) and got != got,
                                f"{name} {label} on {backend}: {got!r}",
                            )
                        else:
                            self.assertEqual(got, reference, f"{name} {label}")

    def test_float64_is_unaffected(self):
        """The change is confined to the binary32 crossing."""
        smallest64 = 5e-324
        for name, function in AFFECTED.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    produced = function(
                        ts.Tensor([smallest64], dtype=ts.float64)
                    ).tolist()[0]
                self.assertNotEqual(
                    produced, 0.0, f"{name} flushed a float64 subnormal"
                )


#: The six kernels whose input conversion was corrected after the first
#: twelve. Only ``log`` failed visibly; see :class:`TheOtherFiveConversions`
#: for why the others could not.
LATER = {
    "log": ts.log,
    "exp": ts.exp,
    "cos": ts.cos,
    "cosh": ts.cosh,
    "arccos": ts.arccos,
    "arccosh": ts.arccosh,
}


def log_of_a_power_of_two(exponent: int) -> float:
    """``log(2**exponent)`` rounded to binary32, by high-precision decimal.

    ``ln(2**n) = n ln 2`` exactly, so the only approximation is ``ln 2``
    itself, which :meth:`decimal.Decimal.ln` produces correctly rounded at
    whatever precision is asked. Fifty digits is far more than the twenty-four
    significand bits of the result need, and nothing here consults a backend
    or a library ``log``.
    """
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 50
        value = Decimal(exponent) * Decimal(2).ln()
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


@requires_cuda
class LogAcceptsSubnormalOperands(unittest.TestCase):
    """``log`` did not merely lose precision: it rejected a valid operand.

    The flushed operand became zero, which is outside ``log``'s domain, so the
    kernel raised ``ValueError`` for an operand the function is defined at.
    This is the one of the six where the output distinguishes a preserved
    operand from a flushed one, and it distinguishes them completely.
    """

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_the_smallest_subnormal_is_accepted(self):
        with ts.use_backend("cuda"):
            produced = ts.log(ts.Tensor([SMALLEST], dtype=ts.float32)).tolist()[0]
        self.assertTrue(math.isfinite(produced), f"log gave {produced!r}")

    def test_every_subnormal_power_of_two_matches_the_decimal_oracle(self):
        """``log(2**-n)`` is ``-n ln 2``; the oracle computes it independently."""
        exponents = list(range(-149, -126))
        operands = [from_bits32(1 << (k + 149)) for k in exponents]
        with ts.use_backend("cuda"):
            produced = ts.log(ts.Tensor(operands, dtype=ts.float32)).tolist()
        for exponent, operand, got in zip(exponents, operands, produced):
            with self.subTest(exponent=exponent):
                self.assertEqual(
                    bits32(got),
                    bits32(log_of_a_power_of_two(exponent)),
                    f"log(2**{exponent}) = log({operand!r}) gave {got!r}",
                )

    def test_the_oracle_agrees_with_exactly_known_logarithms(self):
        """Validate the oracle before it judges anything."""
        self.assertEqual(log_of_a_power_of_two(0), 0.0)
        for exponent in (1, -1, 10, -10, 100):
            with self.subTest(exponent=exponent):
                operand = math.ldexp(1.0, exponent)
                # log(2**n) / log(2) recovers n, within binary32 resolution.
                recovered = log_of_a_power_of_two(exponent) / log_of_a_power_of_two(1)
                self.assertAlmostEqual(recovered, exponent, places=4, msg=repr(operand))

    def test_the_domain_rules_are_unchanged(self):
        """Genuinely invalid operands must still be rejected."""
        for label, value in (
            ("zero", 0.0),
            ("negative zero", -0.0),
            ("negative", -1.0),
            ("negative subnormal", -SMALLEST),
        ):
            for backend in ts.available_backends():
                with self.subTest(case=label, backend=backend):
                    with ts.use_backend(backend):
                        with self.assertRaises(ValueError):
                            ts.log(ts.Tensor([value], dtype=ts.float32))

    def test_a_subnormal_and_zero_are_no_longer_confused(self):
        """The defect made these two operands indistinguishable."""
        with ts.use_backend("cuda"):
            finite = ts.log(ts.Tensor([SMALLEST], dtype=ts.float32)).tolist()[0]
            with self.assertRaises(ValueError):
                ts.log(ts.Tensor([0.0], dtype=ts.float32))
        self.assertTrue(math.isfinite(finite))


@requires_cuda
class TheOtherFiveConversions(unittest.TestCase):
    """``exp``, ``cos``, ``cosh``, ``arccos`` and ``arccosh``.

    None of these can be distinguished by its *output*. Each is even or has a
    derivative of order one at zero, so ``f(subnormal)`` and ``f(0)`` differ by
    at most ``2**-126`` where the result's ulp is at least ``2**-24`` — more
    than a hundred binary orders below half an ulp. The correctly rounded
    results are therefore identical, and no assertion on the returned value
    could tell a preserved operand from a flushed one.

    That is exactly why these five were mistaken for beneficiaries of the
    output-conversion fix. What can be tested is the conversion itself: the
    value that reaches the mathematics must be the value supplied.
    """

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_the_input_conversion_preserves_the_operand(self):
        """The property the output cannot show, asserted where it happens."""
        import cupy

        from tensors.backend.cuda.conversion import _working_values

        operands = [
            SMALLEST,
            -SMALLEST,
            2 * SMALLEST,
            LARGEST_SUBNORMAL,
            -LARGEST_SUBNORMAL,
            MIN_NORMAL,
            0.0,
            -0.0,
            1.5,
        ]
        with ts.use_backend("cuda"):
            tensor = ts.Tensor(operands, dtype=ts.float32)
            widened = cupy.asnumpy(_working_values(tensor))
        for operand, got in zip(operands, widened):
            with self.subTest(operand=operand):
                self.assertEqual(
                    float(got), operand, f"widening changed {operand!r} to {got!r}"
                )

    def test_the_old_conversion_would_have_flushed(self):
        """Pins why the helper is needed, rather than assuming it."""
        import cupy

        with ts.use_backend("cuda"):
            tensor = ts.Tensor([SMALLEST], dtype=ts.float32)
            from tensors.backend.cuda.conversion import tensor_to_logical_array

            flushed = cupy.asnumpy(
                tensor_to_logical_array(tensor).astype(cupy.float64, copy=False)
            )
        self.assertEqual(
            float(flushed[0]), 0.0, "astype no longer flushes; this test is obsolete"
        )

    def test_subnormal_operands_are_accepted_where_the_domain_permits(self):
        for name in ("exp", "cos", "cosh", "arccos"):
            for label, value in (("positive", SMALLEST), ("negative", -SMALLEST)):
                with self.subTest(operation=name, sign=label):
                    with ts.use_backend("cuda"):
                        produced = LATER[name](
                            ts.Tensor([value], dtype=ts.float32)
                        ).tolist()[0]
                    self.assertTrue(
                        math.isfinite(produced), f"{name}({value!r}) = {produced!r}"
                    )

    def test_arccosh_still_rejects_operands_below_one(self):
        """Its domain is [1, inf), so a subnormal is genuinely invalid."""
        for value in (SMALLEST, -SMALLEST, 0.0, 0.5):
            for backend in ts.available_backends():
                with self.subTest(value=value, backend=backend):
                    with ts.use_backend(backend):
                        with self.assertRaises(ValueError):
                            ts.arccosh(ts.Tensor([value], dtype=ts.float32))

    def test_results_stay_in_cuda_storage(self):
        for name, function in LATER.items():
            operand = 1.5 if name == "arccosh" else SMALLEST
            for size in (1, 64, 100_000):
                with self.subTest(operation=name, size=size):
                    with ts.use_backend("cuda"):
                        produced = function(
                            ts.Tensor([operand] * size, dtype=ts.float32)
                        )
                        self.assertEqual(
                            type(produced.backend_storage).__name__, "CudaStorage"
                        )
                        self.assertIs(produced.dtype, ts.float32)

    def test_no_operand_is_read_back_to_the_host(self):
        import contextlib

        import tensors.tensor as tensor_module

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

        for name, function in LATER.items():
            operand = 1.5 if name == "arccosh" else SMALLEST
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    values = ts.Tensor([operand] * 4096, dtype=ts.float32) + 0.0
                    with counting() as reads:
                        produced = function(values)
                        self.assertEqual(
                            type(produced.backend_storage).__name__, "CudaStorage"
                        )
                self.assertEqual(
                    len(reads), 0, f"{name} materialised an operand on the host"
                )

    def test_ordinary_operands_have_not_regressed(self):
        """A divergence here would mean the conversion disturbed a result."""
        for name, function in LATER.items():
            if name == "arccosh":
                values = [1.0, 1.5, 2.0, 10.0]
            elif name == "log":
                values = [0.5, 1.0, 2.0, 10.0]
            elif name == "arccos":
                values = [-0.5, 0.0, 0.5, 1.0]
            else:
                values = [-1.0, -0.5, 0.0, 0.5, 1.0]
            with self.subTest(operation=name):
                results = {}
                for backend in ts.available_backends():
                    with ts.use_backend(backend):
                        results[backend] = function(
                            ts.Tensor(values, dtype=ts.float32)
                        ).tolist()
                for backend, produced in results.items():
                    for index, got in enumerate(produced):
                        self.assertEqual(
                            bits32(got),
                            bits32(results["python"][index]),
                            f"{name} on {backend} at {values[index]!r}",
                        )

    def test_float64_is_unaffected(self):
        for name, function in LATER.items():
            operand = 1.5 if name == "arccosh" else 5e-324
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    produced = function(
                        ts.Tensor([operand], dtype=ts.float64)
                    ).tolist()[0]
                self.assertFalse(math.isnan(produced), name)


#: The two gradient kernels whose input conversion was corrected last.
GRADIENTS = {"abs": ts.abs, "relu": ts.relu}


def vector_jacobian_product(function, operand, upstream, *, dtype, backend="cuda"):
    """One VJP, with the upstream gradient supplied explicitly."""
    with ts.use_backend(backend):
        variable = ts.Variable(ts.Tensor([operand], dtype=dtype), requires_grad=True)
        (produced,) = ts.grad(
            function(variable),
            [variable],
            grad_outputs=ts.Tensor([upstream], dtype=dtype),
        )
        return produced


@requires_cuda
class GradientOperandClassification(unittest.TestCase):
    """The local derivative, from the operand's sign alone.

    ``abs`` and ``relu`` are piecewise linear, so away from zero their
    derivatives are exact integers determined by the sign of the operand:
    ``abs'(x)`` is ``1`` for ``x > 0`` and ``-1`` for ``x < 0``, and
    ``relu'(x)`` is ``1`` for ``x > 0`` and ``0`` for ``x < 0``. A subnormal
    operand is strictly signed like any other, so these follow from the
    definitions and not from any backend.

    The defect flushed the operand to zero, which put every subnormal into
    the ``x == 0`` branch — a misclassification, not a rounding error.
    """

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    #: (operand label, operand, abs derivative, relu derivative)
    SIGNED = (
        ("smallest positive subnormal", SMALLEST, 1.0, 1.0),
        ("smallest negative subnormal", -SMALLEST, -1.0, 0.0),
        ("largest positive subnormal", LARGEST_SUBNORMAL, 1.0, 1.0),
        ("largest negative subnormal", -LARGEST_SUBNORMAL, -1.0, 0.0),
        ("smallest positive normal", MIN_NORMAL, 1.0, 1.0),
        ("smallest negative normal", -MIN_NORMAL, -1.0, 0.0),
        ("ordinary positive", 2.5, 1.0, 1.0),
        ("ordinary negative", -2.5, -1.0, 0.0),
    )

    def test_the_derivative_follows_the_operand_sign(self):
        for label, operand, abs_derivative, relu_derivative in self.SIGNED:
            for name, expected in (("abs", abs_derivative), ("relu", relu_derivative)):
                with self.subTest(operand=label, operation=name):
                    produced = vector_jacobian_product(
                        GRADIENTS[name], operand, 1.0, dtype=ts.float32
                    ).tolist()[0]
                    self.assertEqual(
                        produced, expected, f"{name}'({operand!r}) = {produced!r}"
                    )

    def test_float64_classifies_the_same_way(self):
        smallest64 = 5e-324
        for name, positive, negative in (("abs", 1.0, -1.0), ("relu", 1.0, 0.0)):
            for operand, expected in ((smallest64, positive), (-smallest64, negative)):
                with self.subTest(operation=name, operand=operand):
                    produced = vector_jacobian_product(
                        GRADIENTS[name], operand, 1.0, dtype=ts.float64
                    ).tolist()[0]
                    self.assertEqual(produced, expected)


@requires_cuda
class UpstreamGradientIsPreserved(unittest.TestCase):
    """The second operand the conversion must not lose.

    The operand decides the derivative; the upstream gradient is what that
    derivative scales. They are distinct inputs and were converted by the same
    defective call, so an operand-only correction would leave this half
    broken. These cases use an **ordinary** operand precisely so that a fix
    confined to the operand cannot hide the failure.
    """

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    #: Subnormal upstream gradients, with a local derivative of exactly one,
    #: so the VJP is the upstream gradient itself and is representable.
    UPSTREAM = (
        ("smallest positive subnormal", SMALLEST),
        ("smallest negative subnormal", -SMALLEST),
        ("two quanta", 2 * SMALLEST),
        ("largest subnormal", LARGEST_SUBNORMAL),
        ("largest negative subnormal", -LARGEST_SUBNORMAL),
    )

    def test_a_subnormal_upstream_gradient_survives_a_unit_derivative(self):
        """``f'(2.0) == 1`` for both, so the VJP is exactly the upstream."""
        for name, function in GRADIENTS.items():
            for label, upstream in self.UPSTREAM:
                with self.subTest(operation=name, upstream=label):
                    produced = vector_jacobian_product(
                        function, 2.0, upstream, dtype=ts.float32
                    ).tolist()[0]
                    self.assertEqual(
                        bits32(produced),
                        bits32(upstream),
                        f"{name}: upstream {upstream!r} became {produced!r}",
                    )

    def test_abs_negates_a_subnormal_upstream_at_a_negative_operand(self):
        """``abs'(-2.0) == -1``, so the VJP is the upstream gradient negated."""
        for label, upstream in self.UPSTREAM:
            with self.subTest(upstream=label):
                produced = vector_jacobian_product(
                    ts.abs, -2.0, upstream, dtype=ts.float32
                ).tolist()[0]
                self.assertEqual(
                    bits32(produced),
                    bits32(-upstream),
                    f"upstream {upstream!r} became {produced!r}",
                )

    def test_relu_zeroes_a_subnormal_upstream_at_a_negative_operand(self):
        """``relu'(-2.0) == 0``, so the product is zero however small the upstream."""
        for label, upstream in self.UPSTREAM:
            with self.subTest(upstream=label):
                produced = vector_jacobian_product(
                    ts.relu, -2.0, upstream, dtype=ts.float32
                ).tolist()[0]
                self.assertEqual(produced, 0.0)

    def test_both_operand_and_upstream_subnormal(self):
        """The two conversions together, neither able to mask the other."""
        for name, expected_sign in (("abs", 1.0), ("relu", 1.0)):
            with self.subTest(operation=name):
                produced = vector_jacobian_product(
                    GRADIENTS[name], SMALLEST, SMALLEST, dtype=ts.float32
                ).tolist()[0]
                self.assertEqual(bits32(produced), bits32(SMALLEST), name)


@requires_cuda
class GradientBoundariesAndExistingConventions(unittest.TestCase):
    """Zero and NaN keep the behaviour the package already had."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_zero_and_nan_are_unchanged_across_backends(self):
        """No new derivative convention is introduced at the kink.

        Section 12.7's region-table discipline covers ``**`` only; ``abs``
        and ``relu`` at zero are unspecified (audit finding S-6). What is
        asserted here is that this change did not alter them.
        """
        for name, function in GRADIENTS.items():
            for label, operand in (
                ("positive zero", 0.0),
                ("negative zero", -0.0),
                ("nan", math.nan),
            ):
                with self.subTest(operation=name, operand=label):
                    produced = {}
                    for backend in ts.available_backends():
                        produced[backend] = vector_jacobian_product(
                            function, operand, 1.0, dtype=ts.float32, backend=backend
                        ).tolist()[0]
                    reference = produced["python"]
                    for backend, got in produced.items():
                        if reference != reference:
                            self.assertTrue(got != got, f"{name} {label} {backend}")
                        else:
                            self.assertEqual(got, reference, f"{name} {label}")

    def test_the_largest_finite_upstream_is_unaffected(self):
        largest = 3.4028234663852886e38
        for name, function in GRADIENTS.items():
            with self.subTest(operation=name):
                produced = vector_jacobian_product(
                    function, 2.0, largest, dtype=ts.float32
                ).tolist()[0]
                self.assertEqual(bits32(produced), bits32(largest))

    def test_ordinary_values_have_not_regressed(self):
        """Compared by value, because one zero's *sign* is unspecified.

        See :meth:`test_a_zero_gradients_sign_differs_across_backends`: where
        the result is zero, the backends disagree on its sign, and that
        disagreement predates this change. Comparing by value here keeps this
        test about regressions rather than about an open policy question;
        the sign is examined on its own below.
        """
        for name, function in GRADIENTS.items():
            for operand in (-3.0, -0.5, 0.5, 3.0):
                for upstream in (1.0, -2.0, 0.25):
                    with self.subTest(
                        operation=name, operand=operand, upstream=upstream
                    ):
                        produced = {}
                        for backend in ts.available_backends():
                            produced[backend] = vector_jacobian_product(
                                function,
                                operand,
                                upstream,
                                dtype=ts.float32,
                                backend=backend,
                            ).tolist()[0]
                        for backend, got in produced.items():
                            reference = produced["python"]
                            if reference == 0.0:
                                self.assertEqual(got, 0.0, backend)
                            else:
                                self.assertEqual(
                                    bits32(got), bits32(reference), backend
                                )

    def test_a_zero_gradients_sign_now_agrees_across_backends(self):
        """The open question this test recorded has since been decided.

        It used to assert the divergence: ``relu'(x)`` is zero for
        ``x < 0``, CUDA formed the VJP as ``upstream * derivative``, and
        ``-2.0 * 0.0`` is ``-0.0`` by IEEE while the Python reference
        returned ``+0.0``. Which sign was correct was audit finding
        **S-4**, signed zero being unspecified outside arithmetic.

        docs/relu-semantics.md section 6.1 decides it: the VJP **routes**
        rather than multiplies, so the inactive side is canonical ``+0.0``
        on every backend whatever the upstream is. The test now asserts the
        agreement it once recorded the absence of.
        """
        produced = {}
        for backend in ts.available_backends():
            produced[backend] = vector_jacobian_product(
                ts.relu, -0.5, -2.0, dtype=ts.float32, backend=backend
            ).tolist()[0]

        for backend, got in produced.items():
            with self.subTest(backend=backend):
                self.assertEqual(got, 0.0, f"{backend} gave {got!r}")
                self.assertEqual(
                    math.copysign(1.0, got),
                    1.0,
                    f"{backend} must give canonical +0.0, not -0.0",
                )


@requires_cuda
class GradientExecution(unittest.TestCase):
    """The CUDA kernels run, stay resident, and read nothing back."""

    def setUp(self):
        self.previous = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous)

    def test_the_cuda_gradient_kernels_execute_and_do_not_decline(self):
        import tensors.backend.cuda.kernels as cuda_backend

        for name, function in (
            ("abs_gradient", ts.abs),
            ("relu_gradient", ts.relu),
        ):
            with self.subTest(kernel=name):
                calls = []
                original = getattr(cuda_backend, name)

                def spy(*arguments, _original=original, **keywords):
                    produced = _original(*arguments, **keywords)
                    calls.append(produced is not None)
                    return produced

                with patch.object(cuda_backend, name, spy):
                    with ts.use_backend("cuda"):
                        variable = ts.Variable(
                            ts.Tensor([SMALLEST] * 64, dtype=ts.float32),
                            requires_grad=True,
                        )
                        (produced,) = ts.grad(function(variable), [variable])
                        residency = type(produced.backend_storage).__name__
                self.assertTrue(calls, f"{name} never ran")
                self.assertTrue(all(calls), f"{name} declined to the reference")
                self.assertEqual(residency, "CudaStorage")

    def test_results_stay_in_cuda_storage(self):
        for name, function in GRADIENTS.items():
            for size in (1, 64, 100_000):
                with self.subTest(operation=name, size=size):
                    with ts.use_backend("cuda"):
                        variable = ts.Variable(
                            ts.Tensor([SMALLEST] * size, dtype=ts.float32),
                            requires_grad=True,
                        )
                        (produced,) = ts.grad(function(variable), [variable])
                    self.assertEqual(
                        type(produced.backend_storage).__name__, "CudaStorage"
                    )
                    self.assertIs(produced.dtype, ts.float32)

    def test_no_operand_is_read_back_to_the_host(self):
        import contextlib

        import tensors.tensor as tensor_module

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

        for name, function in GRADIENTS.items():
            with self.subTest(operation=name):
                with ts.use_backend("cuda"):
                    variable = ts.Variable(
                        ts.Tensor([SMALLEST] * 4096, dtype=ts.float32) + 0.0,
                        requires_grad=True,
                    )
                    output = function(variable)
                    with counting() as reads:
                        (produced,) = ts.grad(output, [variable])
                        self.assertEqual(
                            type(produced.backend_storage).__name__, "CudaStorage"
                        )
                self.assertEqual(
                    len(reads), 0, f"{name} materialised a tensor on the host"
                )

    def test_eager_and_graph_differentiation_agree(self):
        from tensors.graph import Computation

        for name, function in GRADIENTS.items():
            for operand in (SMALLEST, -SMALLEST, 2.0):
                with self.subTest(operation=name, operand=operand):
                    with ts.use_backend("cuda"):
                        variable = ts.Variable(
                            ts.Tensor([operand] * 64, dtype=ts.float32),
                            requires_grad=True,
                        )
                        (eager,) = ts.grad(function(variable), [variable])
                        eager_value = eager.tolist()[0]

                        replayed = ts.Variable(
                            ts.Tensor([operand] * 64, dtype=ts.float32),
                            requires_grad=True,
                        )
                        program = Computation(function(replayed))
                        program.forward()
                        (graph,) = ts.grad(function(replayed), [replayed])
                        graph_value = graph.tolist()[0]
                    if eager_value != eager_value:
                        self.assertTrue(graph_value != graph_value)
                    else:
                        self.assertEqual(bits32(graph_value), bits32(eager_value))


if __name__ == "__main__":
    unittest.main()
