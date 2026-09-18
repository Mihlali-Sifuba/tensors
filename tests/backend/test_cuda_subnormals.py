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
                            type(produced._storage).__name__, "CudaStorage"
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
                            type(produced._storage).__name__, "CudaStorage"
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


if __name__ == "__main__":
    unittest.main()
