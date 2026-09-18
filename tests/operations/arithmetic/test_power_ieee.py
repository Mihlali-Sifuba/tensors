"""IEEE floating-point exponentiation (D1 and D2, §§12.2–12.3).

Every expected value is constructed from the table in §12.3.3, written out
below as operand/result triples. No backend's output is used as the source of
an expectation.

Comparison is exact: a zero must carry the specified sign, an infinity must
carry the specified sign, and a NaN is compared by classification only — its
payload and sign are unspecified and are never asserted.

Ordinary finite accuracy is **not** covered here. D5 owns the 2 ULP and 4 ULP
bounds and the validated high-precision reference they are measured against;
nothing in this module establishes them.
"""

import math
import struct
import unittest
from unittest.mock import patch

import tensors as ts

from . import _spec
from ._support import BACKENDS, DTYPE, ArithmeticTestCase, tensor

INF = float("inf")
NAN = float("nan")
FLOATS = _spec.FLOAT_DTYPES


def is_odd_integer(value: float) -> bool:
    """Whether a float is an odd integer; beyond 2**53 every float is even."""
    return value.is_integer() and abs(value) % 2.0 == 1.0


def signbit(value: float) -> bool:
    return math.copysign(1.0, value) < 0.0


def specified_table():
    """Every row of §12.3.3, expanded into (label, base, exponent, result)."""
    rows = []
    add = rows.append
    finite = [0.5, 1.5, 2.0, -0.5, -1.5, -2.0, 3.0, -3.0]

    for base in finite + [0.0, -0.0, INF, -INF, NAN]:
        for exponent in (0.0, -0.0):
            add(("x ** +-0 is 1", base, exponent, 1.0))
    for exponent in finite + [0.0, INF, -INF, NAN]:
        add(("1 ** y is 1", 1.0, exponent, 1.0))
    add(("-1 ** +inf is 1", -1.0, INF, 1.0))
    add(("-1 ** -inf is 1", -1.0, -INF, 1.0))

    for zero in (0.0, -0.0):
        add(("+-0 ** odd > 0", zero, 3.0, math.copysign(0.0, zero)))
        add(("+-0 ** even > 0", zero, 2.0, 0.0))
        add(("+-0 ** non-integral > 0", zero, 0.5, 0.0))
        add(("+-0 ** odd < 0", zero, -3.0, math.copysign(INF, zero)))
        add(("+-0 ** even < 0", zero, -2.0, INF))
        add(("+-0 ** non-integral < 0", zero, -0.5, INF))

    for base in (0.5, -0.5):
        add(("|x| < 1 ** +inf", base, INF, 0.0))
        add(("|x| < 1 ** -inf", base, -INF, INF))
    for base in (2.0, -2.0):
        add(("|x| > 1 ** +inf", base, INF, INF))
        add(("|x| > 1 ** -inf", base, -INF, 0.0))

    add(("-inf ** odd > 0", -INF, 3.0, -INF))
    add(("-inf ** even > 0", -INF, 2.0, INF))
    add(("-inf ** non-integral > 0", -INF, 0.5, INF))
    add(("-inf ** odd < 0", -INF, -3.0, -0.0))
    add(("-inf ** even < 0", -INF, -2.0, 0.0))
    add(("-inf ** non-integral < 0", -INF, -0.5, 0.0))
    add(("+inf ** y > 0", INF, 2.0, INF))
    add(("+inf ** y < 0", INF, -2.0, 0.0))

    for base in (-2.0, -0.5, -8.0):
        for exponent in (0.5, 2.5, 1.0 / 3.0, -0.5):
            add(("negative ** non-integral is NaN", base, exponent, NAN))

    for exponent in (2.0, -2.0, INF, -INF, NAN):
        add(("NaN ** y is NaN", NAN, exponent, NAN))
    for base in (2.0, -2.0, 0.0, INF, -INF):
        add(("x ** NaN is NaN", base, NAN, NAN))
    return tuple(rows)


TABLE = specified_table()


class SpecifiedComparison(ArithmeticTestCase):
    """Exact comparison, with NaN classified rather than equated."""

    def assertSpecified(self, produced, expected, context=""):
        if expected != expected:
            self.assertTrue(
                produced != produced, f"{context}: expected NaN, got {produced!r}"
            )
            return
        self.assertFalse(
            produced != produced, f"{context}: unexpected NaN, wanted {expected!r}"
        )
        self.assertEqual(produced, expected, context)
        if expected == 0.0 or math.isinf(expected):
            self.assertEqual(
                signbit(produced),
                signbit(expected),
                f"{context}: sign of {expected!r} not preserved, got {produced!r}",
            )


class TheSpecialValueTable(SpecifiedComparison):
    """§12.3.3 — every row, both dtypes, every backend."""

    def test_the_table_holds_elementwise(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                for label, base, exponent, expected in TABLE:
                    with self.subTest(
                        dtype=dtype_name,
                        backend=backend,
                        case=label,
                        base=base,
                        exponent=exponent,
                    ):
                        with ts.use_backend(backend):
                            result = tensor(dtype_name, [base]) ** tensor(
                                dtype_name, [exponent]
                            )
                        self.assertIs(result.dtype, DTYPE[dtype_name])
                        self.assertSpecified(result.tolist()[0], expected, f"{label}")

    def test_the_table_holds_for_a_scalar_exponent(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                for label, base, exponent, expected in TABLE:
                    with self.subTest(dtype=dtype_name, backend=backend, case=label):
                        with ts.use_backend(backend):
                            result = tensor(dtype_name, [base]) ** exponent
                        self.assertSpecified(result.tolist()[0], expected, label)

    def test_the_table_holds_for_a_reflected_scalar_base(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                for label, base, exponent, expected in TABLE:
                    if base != base:
                        continue  # S3 admits a NaN literal; covered above
                    with self.subTest(dtype=dtype_name, backend=backend, case=label):
                        with ts.use_backend(backend):
                            result = base ** tensor(dtype_name, [exponent])
                        self.assertSpecified(result.tolist()[0], expected, label)

    def test_a_mixed_array_of_every_category_at_once(self):
        """Finite values, NaNs, infinities and both zeros in one tensor."""
        bases = [r[1] for r in TABLE]
        exponents = [r[2] for r in TABLE]
        expected = [r[3] for r in TABLE]
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        result = (
                            tensor(dtype_name, bases) ** tensor(dtype_name, exponents)
                        ).tolist()
                    for index, (produced, want) in enumerate(zip(result, expected)):
                        self.assertSpecified(produced, want, TABLE[index][0])


class NothingRaises(ArithmeticTestCase):
    """§12.3 — a numerical condition is never an exception."""

    CONDITIONS = (
        ("invalid", [-2.0, -8.0], 0.5),
        ("divide by zero", [0.0, -0.0], -1.0),
        ("overflow", [1e200, -1e200], 2.0),
        ("underflow", [1e-200, -1e-200], 2.0),
        ("infinite operands", [INF, -INF], 0.5),
        ("NaN operands", [NAN, NAN], 2.0),
    )

    def test_no_condition_raises(self):
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                for label, values, exponent in self.CONDITIONS:
                    with self.subTest(dtype=dtype_name, backend=backend, case=label):
                        with ts.use_backend(backend):
                            try:
                                result = tensor(dtype_name, values) ** exponent
                            except (
                                ValueError,
                                OverflowError,
                                ZeroDivisionError,
                            ) as error:
                                self.fail(f"{label} raised {type(error).__name__}")
                        self.assertIs(result.dtype, DTYPE[dtype_name])


class OverflowUnderflowAndSubnormals(SpecifiedComparison):
    """§12.3.4 — overflow to infinity, gradual underflow, signed zero."""

    #: Largest finite value and smallest positive subnormal per dtype.
    LIMITS = {
        "float64": (1.7976931348623157e308, 5e-324, 2.2250738585072014e-308),
        "float32": (
            3.4028234663852886e38,
            1.401298464324817e-45,
            1.1754943508222875e-38,
        ),
    }

    def test_overflow_reaches_infinity(self):
        for dtype_name in FLOATS:
            largest, _, _ = self.LIMITS[dtype_name]
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        result = tensor(dtype_name, [largest, -largest]) ** 2.0
                    produced = result.tolist()
                    self.assertSpecified(produced[0], INF, "positive overflow")
                    self.assertSpecified(produced[1], INF, "negative base squared")

    def test_overflow_keeps_its_sign_for_an_odd_exponent(self):
        for dtype_name in FLOATS:
            largest, _, _ = self.LIMITS[dtype_name]
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        result = tensor(dtype_name, [-largest]) ** 3.0
                    self.assertSpecified(result.tolist()[0], -INF, "odd overflow")

    def test_underflow_is_gradual_and_then_signed_zero(self):
        for dtype_name in FLOATS:
            _, smallest, min_normal = self.LIMITS[dtype_name]
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        # min_normal ** 2 underflows past the subnormal range.
                        deep = tensor(dtype_name, [min_normal, -min_normal]) ** 3.0
                    produced = deep.tolist()
                    self.assertSpecified(produced[0], 0.0, "underflow to +0")
                    self.assertSpecified(produced[1], -0.0, "underflow to -0")

    def test_a_subnormal_is_preserved(self):
        """Gradual underflow: a subnormal operand survives the operation.

        Skipped for `float32` on CUDA. CuPy's generated `float32` code flushes
        subnormals to zero, which section 5.4 records; the four arithmetic
        operations work around it with inline PTX (`mul.rn.f32` and friends),
        but PTX has no `pow` instruction and `powf` flushes internally even
        under `--ftz=false`. `float64` is unaffected on every backend, and
        `float32` holds on Python and NumPy. Closing this needs `powf` built
        from primitives, which belongs with the D5 accuracy work.
        """
        for dtype_name in FLOATS:
            _, smallest, _ = self.LIMITS[dtype_name]
            for backend in BACKENDS:
                if backend == "cuda" and dtype_name == "float32":
                    continue
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        subnormal = tensor(dtype_name, [smallest]) ** 1.0
                    self.assertEqual(subnormal.tolist()[0], smallest)

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_float64_subnormals_survive_on_cuda(self):
        """The dtype the flush does not affect, asserted so the gap is bounded."""
        _, smallest, _ = self.LIMITS["float64"]
        with ts.use_backend("cuda"):
            result = tensor("float64", [smallest]) ** 1.0
        self.assertEqual(result.tolist()[0], smallest)


class RealValuedOnly(SpecifiedComparison):
    """§12.2 — `pow`, not `powr`; strictly real, never complex."""

    def test_negative_base_with_an_integral_exponent_is_valid(self):
        cases = (
            (-2.0, 3.0, -8.0),
            (-2.0, -3.0, -0.125),
            (-2.0, 2.0, 4.0),
            (-2.0, -2.0, 0.25),
            (-3.0, 5.0, -243.0),
        )
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                for base, exponent, expected in cases:
                    with self.subTest(
                        dtype=dtype_name, backend=backend, base=base, exponent=exponent
                    ):
                        with ts.use_backend(backend):
                            result = tensor(dtype_name, [base]) ** exponent
                        self.assertSpecified(result.tolist()[0], expected)

    def test_no_result_is_complex(self):
        for dtype_name in FLOATS:
            with ts.use_backend(BACKENDS[0]):
                result = tensor(dtype_name, [-8.0]) ** (1.0 / 3.0)
            for value in result.tolist():
                self.assertIsInstance(value, float)
                self.assertTrue(math.isnan(value))


class BroadcastingViewsAndVariables(SpecifiedComparison):
    """The table must hold through every operand shape and wrapper."""

    def test_broadcasting(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                grid = ts.Tensor([[0.0, -0.0], [INF, -INF]], dtype=ts.float64)
                column = ts.Tensor([[3.0], [-3.0]], dtype=ts.float64)
                produced = (grid**column).tolist()
                for value, expected in zip(produced, [0.0, -0.0, 0.0, -0.0]):
                    self.assertSpecified(value, expected)

    def test_a_view(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                grid = ts.Tensor([[2.0, 3.0], [0.0, -0.0]], dtype=ts.float64)
                produced = (grid[1] ** -3.0).tolist()
                self.assertSpecified(produced[0], INF)
                self.assertSpecified(produced[1], -INF)

    def test_floating_variables(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend), ts.use_backend(backend):
                base = ts.Variable(
                    tensor("float64", [-2.0, 0.0, INF]), requires_grad=False
                )
                exponent = ts.Variable(
                    tensor("float64", [0.5, -1.0, 2.0]), requires_grad=False
                )
                produced = (base**exponent).data.tolist()
                self.assertTrue(math.isnan(produced[0]))
                self.assertSpecified(produced[1], INF)
                self.assertSpecified(produced[2], INF)


class ReplayAndFusion(SpecifiedComparison):
    """Eager, unfused replay and fused replay must all agree (§8.5)."""

    #: Above the CUDA fusion threshold.
    FUSED_SIZE = 16_384

    def _operands(self, dtype_name, size):
        bases = [r[1] for r in TABLE]
        exponents = [r[2] for r in TABLE]
        repeats = size // len(TABLE) + 1
        return (
            tensor(dtype_name, (bases * repeats)[:size]),
            tensor(dtype_name, (exponents * repeats)[:size]),
            [r[3] for r in TABLE] * repeats,
        )

    def test_unfused_replay_matches_the_table(self):
        from tensors.graph import Computation

        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        base, exponent, expected = self._operands(dtype_name, 64)
                        variables = [
                            ts.Variable(t, requires_grad=False)
                            for t in (base, exponent)
                        ]
                        produced = (
                            Computation(variables[0] ** variables[1]).forward().tolist()
                        )
                    for index, value in enumerate(produced):
                        self.assertSpecified(
                            value, expected[index], TABLE[index % len(TABLE)][0]
                        )

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_fused_replay_matches_eager_and_the_table(self):
        """A second step forces fusion; `* 1.0` preserves signed zeros."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading
        from tensors.graph import Computation

        for dtype_name in FLOATS:
            with self.subTest(dtype=dtype_name), ts.use_backend("cuda"):
                base, exponent, expected = self._operands(dtype_name, self.FUSED_SIZE)
                eager = ((base**exponent) * 1.0).tolist()
                variables = [
                    ts.Variable(t, requires_grad=False) for t in (base, exponent)
                ]
                with patch.object(
                    cuda_backend,
                    "fused_elementwise",
                    wraps=cuda_backend.fused_elementwise,
                ) as fused:
                    loading._clear_backend_kernel_cache()
                    produced = (
                        Computation((variables[0] ** variables[1]) * 1.0)
                        .forward()
                        .tolist()
                    )
                self.assertTrue(fused.called, "the fused kernel was not reached")
            for index, value in enumerate(produced):
                label = TABLE[index % len(TABLE)][0]
                self.assertSpecified(value, expected[index], f"fused: {label}")
                self.assertSpecified(value, eager[index], f"fused vs eager: {label}")

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_a_fused_power_does_not_raise(self):
        from tensors.graph import Computation

        with ts.use_backend("cuda"):
            base = tensor("float64", [-2.0, 0.0, 1e200] * 6000)
            variable = ts.Variable(base, requires_grad=False)
            produced = Computation((variable**0.5) * 1.0).forward().tolist()
        self.assertTrue(math.isnan(produced[0]))
        self.assertEqual(produced[1], 0.0)
        self.assertEqual(produced[2], 1e100)


@unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
class CudaDoesNotTransferOrSynchronise(ArithmeticTestCase):
    """§12.3 — no host synchronisation may detect an exceptional value."""

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

    def test_exceptional_results_stay_on_the_device(self):
        cases = (
            ("all NaN", -2.0, 0.5),
            ("all +inf", 0.0, -1.0),
            ("overflow", 1e200, 2.0),
            ("underflow", 1e-200, 2.0),
        )
        for label, base_value, exponent in cases:
            with self.subTest(case=label):
                with ts.use_backend("cuda"):
                    base = ts.full((4096,), base_value, dtype=ts.float64) + 0.0
                    with self._counting_device_reads() as reads:
                        result = base**exponent
                    self.assertEqual(type(result._storage).__name__, "CudaStorage")
                self.assertEqual(
                    reads.count, 0, f"{label} materialised the tensor on the host"
                )

    def test_the_kernel_answers_every_exceptional_case(self):
        from tensors.backend.loading import _backend_kernel

        for label, base_value, exponent in (
            ("invalid", -2.0, 0.5),
            ("divide by zero", 0.0, -1.0),
            ("overflow", 1e200, 2.0),
        ):
            with self.subTest(case=label):
                with ts.use_backend("cuda"):
                    base = ts.full((64,), base_value, dtype=ts.float64) + 0.0
                    storage = _backend_kernel("power")(
                        base, exponent, dtype=ts.float64, output_shape=(64,)
                    )
                self.assertIsNotNone(storage, f"{label} declined to the reference")
                self.assertEqual(type(storage).__name__, "CudaStorage")


if __name__ == "__main__":
    unittest.main()
