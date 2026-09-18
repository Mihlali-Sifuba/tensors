"""D5 conformance: floating-point exponentiation accuracy (section 12.6).

Every ordinary result must be within 2 ULP (`float64`) or 4 ULP (`float32`) of
the correctly rounded result, **per element**, on every supported execution
path. A single element outside its bound is a failure; an average is never
computed, and no aggregate is allowed to stand in for the per-element test.

Expected values come from :mod:`_reference`, whose own validation is in
:mod:`test_power_reference`. No backend's output is used as an expectation and
no bitwise agreement between backends is required — section 12.6.6 permits two
conforming backends to differ.

The measured maximum for each backend, dtype and execution path is collected
in :data:`MEASUREMENTS` and asserted to be complete, so a configuration cannot
pass by never having been run.
"""

from __future__ import annotations

import math
import struct
import unittest
from functools import lru_cache
from unittest.mock import patch

import tensors as ts

from . import _accuracy, _power_cases
from ._reference import (
    OutsideReferenceDomain,
    UnresolvedReference,
    reference_power,
)
from ._support import BACKENDS, DTYPE, ArithmeticTestCase

FLOATS = ("float32", "float64")

#: Above the CUDA fusion threshold.
FUSED_SIZE = 16_384

#: (backend, dtype, path) -> (maximum ULP observed, elements compared).
MEASUREMENTS: dict[tuple[str, str, str], tuple[int, int]] = {}


@lru_cache(maxsize=None)
def _reference_value(base: float, exponent: float, dtype_name: str):
    """The correctly rounded result, or why the metric does not apply."""
    try:
        return reference_power(base, exponent, dtype_name), None
    except OutsideReferenceDomain as exc:
        return None, str(exc)


class AccuracyHarness(ArithmeticTestCase):
    """Runs operand pairs through a path and assesses every element."""

    def assess(self, *, backend, dtype_name, path, bases, exponents, produced):
        """Compare each element, and record the configuration's maximum.

        ``bases`` and ``exponents`` are read back out of the tensors, so the
        reference is asked about exactly the values the kernel evaluated —
        including a scalar that rule S3 has already converted.
        """
        self.assertEqual(len(produced), len(bases))
        worst = 0
        compared = 0
        failures = []
        unresolved = []
        for index, value in enumerate(produced):
            base, exponent = bases[index], exponents[index]
            try:
                reference, skip = _reference_value(base, exponent, dtype_name)
            except UnresolvedReference as exc:
                unresolved.append(str(exc))
                continue
            if reference is None:
                continue  # a special value; section 12.6.1 and D1/D2 own it
            comparison = _accuracy.compare(value, reference.value, dtype_name)
            compared += 1
            if comparison.distance is not None:
                worst = max(worst, comparison.distance)
            if not comparison.conforms:
                failures.append(
                    _accuracy.describe(
                        base=base,
                        exponent=exponent,
                        dtype=dtype_name,
                        backend=backend,
                        mode=path,
                        produced=value,
                        expected=reference.value,
                        comparison=comparison,
                        reference=reference.describe(),
                    )
                )

        key = (backend, dtype_name, path)
        previous = MEASUREMENTS.get(key, (0, 0))
        MEASUREMENTS[key] = (max(previous[0], worst), previous[1] + compared)

        # assertFalse rather than assertEqual against []: unittest's sequence
        # diff truncates, and a diagnostic that cannot be read is not one.
        self.assertFalse(
            unresolved,
            "the reference could not resolve a case; it must not be guessed:\n"
            + "\n".join(unresolved[:5]),
        )
        misclassified = sum(1 for entry in failures if "no ULP distance" in entry)
        self.assertFalse(
            failures,
            f"{len(failures)} of {compared} elements do not conform on "
            f"{backend}/{path}/{dtype_name}: {misclassified} are the wrong "
            f"class of value and {len(failures) - misclassified} exceed the "
            f"{_accuracy.BOUNDS[dtype_name]} ULP bound. The largest distance "
            f"among the elements the metric does apply to is {worst} ULP.\n\n"
            + "\n\n".join(failures[:8]),
        )
        return worst

    # -- the execution paths --------------------------------------------

    def run_eager_tensor_tensor(self, backend, dtype_name, cases):
        with ts.use_backend(backend):
            base = ts.Tensor([c[1] for c in cases], dtype=DTYPE[dtype_name])
            exponent = ts.Tensor([c[2] for c in cases], dtype=DTYPE[dtype_name])
            produced = (base**exponent).tolist()
            return base.tolist(), exponent.tolist(), produced

    def run_eager_scalar(self, backend, dtype_name, cases, *, reflected):
        """One element per call, since a scalar operand carries one value."""
        bases, exponents, produced = [], [], []
        with ts.use_backend(backend):
            for _, base, exponent in cases:
                if reflected:
                    operand = ts.Tensor([exponent], dtype=DTYPE[dtype_name])
                    produced.append((base**operand).tolist()[0])
                    bases.append(base)
                    exponents.append(operand.tolist()[0])
                else:
                    operand = ts.Tensor([base], dtype=DTYPE[dtype_name])
                    produced.append((operand**exponent).tolist()[0])
                    bases.append(operand.tolist()[0])
                    exponents.append(exponent)
        return bases, exponents, produced

    def run_replay(self, backend, dtype_name, cases):
        from tensors.graph import Computation

        with ts.use_backend(backend):
            base = ts.Tensor([c[1] for c in cases], dtype=DTYPE[dtype_name])
            exponent = ts.Tensor([c[2] for c in cases], dtype=DTYPE[dtype_name])
            left = ts.Variable(base, requires_grad=False)
            right = ts.Variable(exponent, requires_grad=False)
            produced = Computation(left**right).forward().tolist()
            return base.tolist(), exponent.tolist(), produced

    def run_fused(self, dtype_name, cases):
        """A second step forces fusion; the kernel is asserted to be reached."""
        import tensors.backend.cuda.kernels as cuda_backend
        from tensors.backend import loading
        from tensors.graph import Computation

        count = len(cases)
        repeats = FUSED_SIZE // count + 1
        with ts.use_backend("cuda"):
            base = ts.Tensor([c[1] for c in cases] * repeats, dtype=DTYPE[dtype_name])
            exponent = ts.Tensor(
                [c[2] for c in cases] * repeats, dtype=DTYPE[dtype_name]
            )
            left = ts.Variable(base, requires_grad=False)
            right = ts.Variable(exponent, requires_grad=False)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fused:
                loading._clear_backend_kernel_cache()
                produced = Computation((left**right) * 1.0).forward().tolist()
            self.assertTrue(fused.called, "the fused CUDA kernel was not reached")
        return (
            base.tolist()[:count],
            exponent.tolist()[:count],
            produced[:count],
        )


class ConstructedCases(AccuracyHarness):
    """The reasoned boundary cases, on every path."""

    def cases(self, dtype_name):
        return _power_cases.constructed(dtype_name)

    def test_eager_tensor_tensor(self):
        for dtype_name in FLOATS:
            cases = self.cases(dtype_name)
            for backend in BACKENDS:
                with self.subTest(backend=backend, dtype=dtype_name):
                    bases, exponents, produced = self.run_eager_tensor_tensor(
                        backend, dtype_name, cases
                    )
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="eager tensor ** tensor",
                        bases=bases,
                        exponents=exponents,
                        produced=produced,
                    )

    def test_eager_tensor_scalar(self):
        for dtype_name in FLOATS:
            cases = self.cases(dtype_name)
            for backend in BACKENDS:
                with self.subTest(backend=backend, dtype=dtype_name):
                    bases, exponents, produced = self.run_eager_scalar(
                        backend, dtype_name, cases, reflected=False
                    )
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="eager tensor ** scalar",
                        bases=bases,
                        exponents=exponents,
                        produced=produced,
                    )

    def test_eager_reflected_scalar_base(self):
        for dtype_name in FLOATS:
            cases = self.cases(dtype_name)
            for backend in BACKENDS:
                with self.subTest(backend=backend, dtype=dtype_name):
                    bases, exponents, produced = self.run_eager_scalar(
                        backend, dtype_name, cases, reflected=True
                    )
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="eager scalar ** tensor",
                        bases=bases,
                        exponents=exponents,
                        produced=produced,
                    )

    def test_unfused_graph_replay(self):
        for dtype_name in FLOATS:
            cases = self.cases(dtype_name)
            for backend in BACKENDS:
                with self.subTest(backend=backend, dtype=dtype_name):
                    bases, exponents, produced = self.run_replay(
                        backend, dtype_name, cases
                    )
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="graph replay",
                        bases=bases,
                        exponents=exponents,
                        produced=produced,
                    )

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_fused_cuda_replay(self):
        for dtype_name in FLOATS:
            cases = self.cases(dtype_name)
            with self.subTest(dtype=dtype_name):
                bases, exponents, produced = self.run_fused(dtype_name, cases)
                self.assess(
                    backend="cuda",
                    dtype_name=dtype_name,
                    path="fused replay",
                    bases=bases,
                    exponents=exponents,
                    produced=produced,
                )


class SampledCases(AccuracyHarness):
    """Deterministic random coverage, supplementing the constructed cases."""

    def test_eager_tensor_tensor(self):
        for dtype_name in FLOATS:
            cases = _power_cases.sampled(dtype_name)
            for backend in BACKENDS:
                with self.subTest(backend=backend, dtype=dtype_name):
                    bases, exponents, produced = self.run_eager_tensor_tensor(
                        backend, dtype_name, cases
                    )
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="eager tensor ** tensor",
                        bases=bases,
                        exponents=exponents,
                        produced=produced,
                    )

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_fused_cuda_replay(self):
        for dtype_name in FLOATS:
            cases = _power_cases.sampled(dtype_name)
            with self.subTest(dtype=dtype_name):
                bases, exponents, produced = self.run_fused(dtype_name, cases)
                self.assess(
                    backend="cuda",
                    dtype_name=dtype_name,
                    path="fused replay",
                    bases=bases,
                    exponents=exponents,
                    produced=produced,
                )


class ShapesLayoutsAndMixedDtypes(AccuracyHarness):
    """The bound holds whatever the shape, the layout or the promotion."""

    PAIRS = (
        (1.5, 2.5),
        (0.75, -1.5),
        (3.0, 0.3333333333333333),
        (2.0, 0.5),
        (10.0, 1.7),
        (0.1, 3.3),
        (-2.5, 3.0),
        (7.0, -0.25),
    )

    def test_several_tensor_sizes(self):
        """Small tensors and sizes above the fusion threshold alike."""
        for size in (1, 2, 3, 7, 64, 1000, FUSED_SIZE + 5):
            cases = [("size", *self.PAIRS[i % len(self.PAIRS)]) for i in range(size)]
            for dtype_name in FLOATS:
                for backend in BACKENDS:
                    with self.subTest(size=size, dtype=dtype_name, backend=backend):
                        bases, exponents, produced = self.run_eager_tensor_tensor(
                            backend, dtype_name, cases
                        )
                        self.assess(
                            backend=backend,
                            dtype_name=dtype_name,
                            path="eager tensor ** tensor",
                            bases=bases,
                            exponents=exponents,
                            produced=produced,
                        )

    def test_broadcasting(self):
        bases = [c[0] for c in self.PAIRS]
        exponents = [c[1] for c in self.PAIRS]
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        left = ts.Tensor(
                            [[value] for value in bases], dtype=DTYPE[dtype_name]
                        )
                        right = ts.Tensor([exponents], dtype=DTYPE[dtype_name])
                        result = left**right
                        self.assertEqual(result.shape, (len(bases), len(exponents)))
                        # tolist is flat and row-major, so the pairing below
                        # is the order the elements come back in.
                        produced = result.tolist()
                        stored_bases = left.tolist()
                        stored_exponents = right.tolist()
                    flat_bases, flat_exponents = [], []
                    for base in stored_bases:
                        for exponent in stored_exponents:
                            flat_bases.append(base)
                            flat_exponents.append(exponent)
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="broadcast",
                        bases=flat_bases,
                        exponents=flat_exponents,
                        produced=produced,
                    )

    def test_a_strided_view(self):
        """Every other element of a larger tensor, so the layout is not dense."""
        for dtype_name in FLOATS:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        pairs = [self.PAIRS[i % len(self.PAIRS)] for i in range(64)]
                        base = ts.Tensor(
                            [p[0] for p in pairs], dtype=DTYPE[dtype_name]
                        )[::2]
                        exponent = ts.Tensor(
                            [p[1] for p in pairs], dtype=DTYPE[dtype_name]
                        )[::2]
                        produced = (base**exponent).tolist()
                        stored_bases = base.tolist()
                        stored_exponents = exponent.tolist()
                    self.assess(
                        backend=backend,
                        dtype_name=dtype_name,
                        path="strided view",
                        bases=stored_bases,
                        exponents=stored_exponents,
                        produced=produced,
                    )

    def test_mixed_floating_dtypes_promote_and_still_conform(self):
        """``float32 ** float64`` promotes to ``float64`` and is bounded there."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    base = ts.Tensor([p[0] for p in self.PAIRS], dtype=ts.float32)
                    exponent = ts.Tensor([p[1] for p in self.PAIRS], dtype=ts.float64)
                    produced = base**exponent
                    self.assertIs(produced.dtype, ts.float64)
                    stored_bases = [float(v) for v in base.tolist()]
                    stored_exponents = exponent.tolist()
                    values = produced.tolist()
                self.assess(
                    backend=backend,
                    dtype_name="float64",
                    path="mixed dtypes",
                    bases=stored_bases,
                    exponents=stored_exponents,
                    produced=values,
                )


class ThereIsNoUnmeasuredPath(ArithmeticTestCase):
    """Every path that can evaluate a power has been measured."""

    def test_the_python_and_numpy_fused_kernels_decline_a_power(self):
        """Only CUDA fuses a power, so only CUDA has a fused path to measure."""
        from tensors.backend.python.kernels.fusion.fused_elementwise import (
            fused_elementwise as python_fused,
        )

        steps = (("power", None, False, 1), ("multiply", None, False, 2))
        with ts.use_backend("python"):
            values = (
                ts.Tensor([2.0], dtype=ts.float32),
                ts.Tensor([3.0], dtype=ts.float32),
                ts.Tensor([1.0], dtype=ts.float32),
            )
            self.assertIsNone(
                python_fused(values, steps, dtype=ts.float32, output_shape=(1,)),
                "the Python fused kernel accepted a power; it is unmeasured",
            )

        if "numpy" in BACKENDS:
            from tensors.backend.numpy.kernels.fusion.fused_elementwise import (
                fused_elementwise as numpy_fused,
            )

            with ts.use_backend("numpy"):
                values = (
                    ts.Tensor([2.0], dtype=ts.float32),
                    ts.Tensor([3.0], dtype=ts.float32),
                    ts.Tensor([1.0], dtype=ts.float32),
                )
                self.assertIsNone(
                    numpy_fused(values, steps, dtype=ts.float32, output_shape=(1,)),
                    "the NumPy fused kernel accepted a power; it is unmeasured",
                )

    def test_every_configuration_was_measured(self):
        """A configuration cannot conform by never having been run."""
        expected = set()
        for backend in BACKENDS:
            for dtype_name in FLOATS:
                for path in (
                    "eager tensor ** tensor",
                    "eager tensor ** scalar",
                    "eager scalar ** tensor",
                    "graph replay",
                ):
                    expected.add((backend, dtype_name, path))
        if "cuda" in BACKENDS:
            for dtype_name in FLOATS:
                expected.add(("cuda", dtype_name, "fused replay"))

        missing = expected - set(MEASUREMENTS)
        self.assertEqual(
            missing,
            set(),
            "no accuracy measurement was recorded for: " + repr(sorted(missing)),
        )
        for key, (worst, compared) in sorted(MEASUREMENTS.items()):
            with self.subTest(configuration=key):
                self.assertGreater(compared, 0, f"{key} compared no elements")
                self.assertLessEqual(
                    worst,
                    _accuracy.BOUNDS[key[1]],
                    f"{key}: {worst} ULP over {compared} elements",
                )


class TheWidenAndNarrowStrategy(unittest.TestCase):
    """What evaluating a binary32 power in binary64 can cost (section 12.6.2).

    All three backends now evaluate a binary32 power in binary64 and round
    once to binary32 — CUDA through PTX conversions, NumPy through a widened
    array, and the Python backend implicitly, since ``math.pow`` is binary64
    and the result is stored into a binary32 array.

    That is a double rounding, and these tests bound what it can cost rather
    than relying on the measurements alone. Sampling cannot reach the regime
    where it bites: a binary64 ``pow`` carries a relative error of order
    2**-52, so the two roundings disagree only when the true value lies within
    about that distance of a binary32 rounding boundary, and a random search
    would need of the order of 2**36 candidates to find one.
    """

    def test_a_double_rounding_disagreement_costs_exactly_one_ulp(self):
        """Constructed, since it cannot be found by sampling."""
        from fractions import Fraction

        from ._reference import round_fraction

        midpoint = Fraction(1) + Fraction(*(2.0**-24).as_integer_ratio())
        value = midpoint + Fraction(1, 2**80)

        once = round_fraction(value, "float32")
        through = round_fraction(
            Fraction(*round_fraction(value, "float64").as_integer_ratio()), "float32"
        )

        self.assertNotEqual(once, through, "the construction did not disagree")
        distance = abs(
            _accuracy.monotone(once, "float32") - _accuracy.monotone(through, "float32")
        )
        self.assertEqual(distance, 1, "a double rounding moved more than one ULP")
        self.assertLessEqual(distance, _accuracy.BOUNDS["float32"])

    def test_the_binary64_error_is_far_below_a_binary32_ulp(self):
        """The premise of the argument, stated as arithmetic.

        One binary32 ULP is 2**-23 relative; the binary64 error the vendors
        document for ``pow`` is a couple of 2**-52. The ratio is what makes
        the strategy sound.
        """
        binary32_ulp = 2.0**-23
        binary64_error = 2 * 2.0**-52
        self.assertLess(binary64_error / binary32_ulp, 2.0**-27)


class TheMetricItself(unittest.TestCase):
    """Section 12.6.4, checked directly rather than only through the kernels."""

    def successor(self, value, dtype_name):
        """The next representable value *of that format*, not of binary64."""
        if dtype_name == "float64":
            return math.nextafter(value, math.inf)
        pattern = struct.unpack("<I", struct.pack("<f", value))[0]
        return struct.unpack("<f", struct.pack("<I", pattern + 1))[0]

    def test_neighbours_differ_by_one(self):
        for dtype_name in FLOATS:
            for value in (1.0, 1e10, 1e-10, 1.1754943508222875e-38, 1e-44):
                with self.subTest(dtype=dtype_name, value=value):
                    self.assertEqual(
                        _accuracy.monotone(
                            self.successor(value, dtype_name), dtype_name
                        )
                        - _accuracy.monotone(value, dtype_name),
                        1,
                    )

    def test_the_mapping_crosses_the_subnormal_boundary(self):
        smallest_normal = 1.1754943508222875e-38
        largest_subnormal = 1.1754942106924411e-38
        self.assertEqual(
            _accuracy.monotone(smallest_normal, "float32")
            - _accuracy.monotone(largest_subnormal, "float32"),
            1,
        )

    def test_the_mapping_is_monotone_through_zero(self):
        ordered = [-1e-45, -0.0, 0.0, 1e-45]
        mapped = [_accuracy.monotone(v, "float32") for v in ordered]
        self.assertEqual(mapped, sorted(mapped))

    def test_nan_is_compared_by_classification(self):
        nan = float("nan")
        self.assertTrue(_accuracy.compare(nan, nan, "float64").conforms)
        self.assertIsNone(_accuracy.compare(nan, nan, "float64").distance)
        self.assertFalse(_accuracy.compare(nan, 1.0, "float64").conforms)
        self.assertFalse(_accuracy.compare(1.0, nan, "float64").conforms)

    def test_signed_zero_is_exact(self):
        self.assertTrue(_accuracy.compare(0.0, 0.0, "float64").conforms)
        self.assertTrue(_accuracy.compare(-0.0, -0.0, "float64").conforms)
        self.assertFalse(_accuracy.compare(0.0, -0.0, "float64").conforms)
        self.assertIsNone(_accuracy.compare(0.0, -0.0, "float64").distance)

    def test_a_zero_against_a_tiny_finite_is_a_failure_not_a_distance(self):
        comparison = _accuracy.compare(0.0, 5e-324, "float64")
        self.assertFalse(comparison.conforms)
        self.assertIsNone(comparison.distance)

    def test_infinities_are_exact(self):
        infinity = float("inf")
        self.assertTrue(_accuracy.compare(infinity, infinity, "float64").conforms)
        self.assertFalse(_accuracy.compare(infinity, -infinity, "float64").conforms)
        self.assertFalse(
            _accuracy.compare(infinity, 1.7976931348623157e308, "float64").conforms
        )
        self.assertIsNone(
            _accuracy.compare(infinity, 1.7976931348623157e308, "float64").distance
        )

    def test_opposite_signs_fail_without_a_distance(self):
        comparison = _accuracy.compare(1.0, -1.0, "float64")
        self.assertFalse(comparison.conforms)
        self.assertIsNone(comparison.distance)

    def test_the_bound_is_inclusive(self):
        value = 1.0
        two_away = math.nextafter(math.nextafter(value, math.inf), math.inf)
        three_away = math.nextafter(two_away, math.inf)
        self.assertTrue(_accuracy.compare(two_away, value, "float64").conforms)
        self.assertEqual(_accuracy.compare(two_away, value, "float64").distance, 2)
        self.assertFalse(_accuracy.compare(three_away, value, "float64").conforms)
        self.assertEqual(_accuracy.compare(three_away, value, "float64").distance, 3)

    def test_the_bounds_are_the_specified_ones(self):
        self.assertEqual(_accuracy.BOUNDS, {"float64": 2, "float32": 4})


if __name__ == "__main__":
    unittest.main()
