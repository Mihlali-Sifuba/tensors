"""Where forward power executes, and what a provider decline still means.

`docs/backends.md`, *Execution requirements*: the backend selection decides
where an operation runs. Forward power used to consult a workload-size
threshold, so a small exponentiation ran in Python whatever was selected.
These tests ask where it ran, by storage kind and by provider call count,
rather than by the numbers coming out.

The decline path is deliberately still here, and the last class pins down why:
power's kernels use ``None`` for a domain error, for CUDA's missing integer
exponentiation, and for a narrowing result, and the reference is what turns
each of those into the documented answer. Execution location is therefore
guaranteed for the supported cases only. Power's backward pass is untouched
and is not covered here.
"""

import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
import tensors.backend.numpy.kernels as numpy_kernels
from tensors.backend import config
from tests.backend._support import requires_cuda, requires_numpy

STORAGE_FOR = {
    "python": "PythonStorage",
    "numpy": "NumPyStorage",
    "cuda": "CudaStorage",
}

#: Below and above the 32-element threshold this change removed.
SIZES = (1, 4, 31, 32, 64, 1000)


def storage_name(tensor) -> str:
    return type(tensor._storage).__name__


def power_storages(selection, size):
    """Exponentiate both ways round and report where each result landed."""
    with ts.use_backend(selection):
        base = ts.full((size,), 2.0, dtype=ts.float64)
        exponent = ts.full((size,), 3.0, dtype=ts.float64)
        return (
            storage_name(base**2.0),  # tensor ** scalar
            storage_name(base**exponent),  # tensor ** tensor
            storage_name(2.0**exponent),  # scalar ** tensor
        )


class ForwardPowerExecutesOnTheSelectedBackend(unittest.TestCase):
    """Every size, and all three ways of writing an exponentiation."""

    def _assert_runs_on(self, selection, expected=None):
        expected = STORAGE_FOR[expected or selection]
        for size in SIZES:
            with self.subTest(size=size):
                self.assertEqual(
                    power_storages(selection, size), (expected, expected, expected)
                )

    def test_explicit_python_executes_on_python(self):
        self._assert_runs_on("python")

    @requires_numpy
    def test_explicit_numpy_executes_on_numpy_including_small_work(self):
        self._assert_runs_on("numpy")

    @requires_cuda
    def test_explicit_cuda_executes_on_cuda_including_small_work(self):
        self._assert_runs_on("cuda")

    @requires_numpy
    def test_automatic_selection_with_numpy_executes_on_numpy(self):
        with ts.use_backend("auto"):
            self.assertEqual(ts.get_backend(), "numpy")
        self._assert_runs_on("auto", expected="numpy")

    def test_automatic_selection_without_numpy_executes_on_python(self):
        """``auto`` falls back to Python when NumPy is not installed.

        Everything happens inside one patched block. The simulated absence is
        asserted first, so the test fails loudly if the patch ever stops
        reaching :func:`_resolve_backend`, rather than quietly measuring a
        machine that has NumPy after all. Resolution and execution are checked
        in the same context for the same reason: a patch that covered only the
        resolution check would leave the exponentiation running on NumPy.
        """
        with patch.object(config, "_numpy_available", return_value=False):
            self.assertNotIn("numpy", ts.available_backends())

            with ts.use_backend("auto"):
                self.assertEqual(ts.get_backend(), "python")

                for size in SIZES:
                    with self.subTest(size=size):
                        base = ts.full((size,), 2.0, dtype=ts.float64)
                        exponent = ts.full((size,), 3.0, dtype=ts.float64)
                        for result in (
                            base**2.0,  # tensor ** scalar
                            base**exponent,  # tensor ** tensor
                            2.0**exponent,  # scalar ** tensor
                        ):
                            self.assertEqual(storage_name(result), "PythonStorage")


@requires_numpy
class SmallPowerReachesTheProviderKernel(unittest.TestCase):
    """Called, rather than skipped because the workload looked small."""

    def test_a_four_element_power_calls_the_numpy_kernel(self):
        with patch.object(numpy_kernels, "power", wraps=numpy_kernels.power) as kernel:
            backend_state._clear_backend_kernel_cache()
            with ts.use_backend("numpy"):
                base = ts.full((4,), 2.0, dtype=ts.float64)
                _ = base**2.0
                _ = base**base
                _ = 2.0**base
        backend_state._clear_backend_kernel_cache()
        self.assertEqual(kernel.call_count, 3)

    def test_a_single_element_power_calls_the_numpy_kernel(self):
        with patch.object(numpy_kernels, "power", wraps=numpy_kernels.power) as kernel:
            backend_state._clear_backend_kernel_cache()
            with ts.use_backend("numpy"):
                _ = ts.full((1,), 2.0, dtype=ts.float64) ** 2.0
        backend_state._clear_backend_kernel_cache()
        self.assertEqual(kernel.call_count, 1)


class ResultsAndErrorsAreUnchanged(unittest.TestCase):
    """Relocating the work moved no value and no exception.

    Integer results and every dtype are compared exactly. Floating results
    are compared closely rather than bitwise, because power with a
    non-integral exponent has never agreed to the last bit across backends —
    ``9.0 ** 0.5`` is 3.0 in Python and 2.9999999999999996 on CUDA, and that
    predates this change.
    """

    SUPPORTED = (
        ("float64 square", [2.0, -3.0, 0.5], 2.0, ts.float64),
        ("float64 cube", [2.0, -3.0, 0.5], 3.0, ts.float64),
        ("float32 square", [2.0, -3.0, 0.5], 2.0, ts.float32),
        ("int32 cube", [2, -3, 7], 3, ts.int32),
        ("int64 zeroth", [2, -3, 7], 0, ts.int64),
        ("uint8 square", [2, 3, 7], 2, ts.uint8),
        ("negative exponent", [2.0, 4.0, 0.5], -1.0, ts.float64),
        ("fractional exponent", [4.0, 9.0, 16.0], 0.5, ts.float64),
    )

    DOMAIN_ERRORS = (
        ("negative base, fractional exponent", [-2.0, -3.0], 0.5, ts.float64),
        ("zero base, negative exponent", [0.0, 0.0], -1.0, ts.float64),
        ("float64 overflow", [1e200, 1e200], 2.0, ts.float64),
    )

    def _each_size(self):
        return (1, 4, 31, 64)

    def test_supported_results_match_the_python_backend(self):
        for label, values, exponent, dtype in self.SUPPORTED:
            for size in self._each_size():
                data = (values * (size // len(values) + 1))[:size]
                with ts.use_backend("python"):
                    expected = (ts.Tensor(data, dtype=dtype) ** exponent).tolist()
                for backend in ts.available_backends():
                    with self.subTest(case=label, size=size, backend=backend):
                        with ts.use_backend(backend):
                            result = ts.Tensor(data, dtype=dtype) ** exponent
                        self.assertIs(result.dtype, expected_dtype(dtype, exponent))
                        produced = result.tolist()
                        self.assertEqual(len(produced), len(expected))
                        if result.dtype.kind == "integer":
                            self.assertEqual(produced, expected)
                            continue
                        for got, want in zip(produced, expected):
                            self.assertAlmostEqual(got, want, places=6)

    def test_exceptional_values_are_delivered_not_raised(self):
        """Breaking changes B17, B18 and B19.

        These three cases raised `ValueError` or `OverflowError`. §12.3 makes
        each of them a value, identical on every backend. The exhaustive table
        lives in `tests/operations/arithmetic/test_power_ieee.py`; this checks
        that the execution paths here deliver it rather than raising.
        """
        import math

        expected = {
            "negative base, fractional exponent": math.isnan,
            "zero base, negative exponent": lambda v: v == math.inf,
            "float64 overflow": lambda v: v == math.inf,
        }
        for label, values, exponent, dtype in self.DOMAIN_ERRORS:
            for size in self._each_size():
                data = (values * (size // len(values) + 1))[:size]
                for backend in ts.available_backends():
                    with self.subTest(case=label, size=size, backend=backend):
                        with ts.use_backend(backend):
                            result = ts.Tensor(data, dtype=dtype) ** exponent
                        self.assertIs(result.dtype, dtype)
                        self.assertTrue(
                            all(expected[label](v) for v in result.tolist()),
                            f"{label} did not deliver its specified value",
                        )

    def test_a_scalar_base_over_an_integer_exponent_still_works(self):
        for backend in ts.available_backends():
            for size in self._each_size():
                with self.subTest(backend=backend, size=size):
                    with ts.use_backend(backend):
                        exponent = ts.Tensor([0, 1, 2, 3] * size, dtype=ts.int32)
                        result = 2**exponent
                    self.assertIs(result.dtype, ts.int32)
                    self.assertEqual(result.tolist(), [1, 2, 4, 8] * size)


def expected_dtype(dtype, exponent):
    """The dtype §12.5 gives for this base dtype and exponent."""
    from tensors.dtype import resolve_power

    return resolve_power(dtype, exponent)[0]


@requires_numpy
class TheProviderNoLongerDeclines(unittest.TestCase):
    """What used to be power's decline path.

    Power's kernels returned ``None`` for three different things: a domain
    error, CUDA's missing integer exponentiation, and an integer result that
    left the declared dtype. D1 to D4 removed all three — domain violations
    are values (§12.3), integer exponentiation is native and wraps (§12.4.1),
    and CUDA computes integers itself. The kernel is therefore expected to
    answer every supported call.

    The dispatcher still routes a ``None`` to the Python reference. Making
    that a hard error belongs to the strict-dispatch milestone, so this class
    records the behaviour rather than asserting it is unreachable.
    """

    def test_the_kernel_answers_instead_of_declining(self):
        for label, values, exponent, dtype in (
            ("negative base, fractional exponent", [-2.0], 0.5, ts.float64),
            ("zero base, negative exponent", [0.0], -1.0, ts.float64),
            ("float64 overflow", [1e200], 2.0, ts.float64),
            ("integer overflow", [2], 31, ts.int32),
        ):
            with self.subTest(case=label):
                with ts.use_backend("numpy"):
                    base = ts.Tensor(values * 64, dtype=dtype)
                    storage = numpy_kernels.power(
                        base, exponent, dtype=dtype, output_shape=(64,)
                    )
                self.assertIsNotNone(
                    storage, f"{label} still declines to the reference"
                )

    def test_a_patched_decline_is_still_answered_by_the_reference(self):
        """The dispatcher's fallback, exercised by forcing a decline."""
        with patch.object(numpy_kernels, "power", return_value=None) as declining:
            backend_state._clear_backend_kernel_cache()
            with ts.use_backend("numpy"):
                result = ts.full((64,), 2.0, dtype=ts.float64) ** 3.0
            backend_state._clear_backend_kernel_cache()
        declining.assert_called_once()
        self.assertEqual(storage_name(result), "PythonStorage")
        self.assertEqual(result.tolist(), [8.0] * 64)


if __name__ == "__main__":
    unittest.main()
