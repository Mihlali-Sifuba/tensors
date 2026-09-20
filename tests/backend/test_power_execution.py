"""Where forward power executes, and what a provider decline means.

`docs/backends.md`, *Execution requirements*: the backend selection decides
where an operation runs, and a provider that cannot execute an operation says
so rather than handing it to another backend. These tests ask where the work
ran — by provider call count, by the Python reference *not* being called, and
by storage residency — rather than by the numbers coming out, because a
correct answer computed in the wrong place is the defect being tested for.

Power's kernels used ``None`` for three different things: a domain error,
CUDA's missing integer exponentiation, and a narrowing integer result. D1 to
D4 removed all three, so a decline now means a capability gap and the
dispatcher raises. Power's backward pass has its own strict dispatchers and
is covered by the D7 tests, not here.
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


def python_reference():
    """The Python power kernel's module, for instrumenting fallbacks.

    Imported by name: the package re-exports the function under the same
    name as its module, so a plain ``import ... as`` binds the function.
    """
    import importlib

    return importlib.import_module("tensors.backend.python.kernels.arithmetic.power")


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
        reaching :func:`resolve_backend`, rather than quietly measuring a
        machine that has NumPy after all. Resolution and execution are checked
        in the same context for the same reason: a patch that covered only the
        resolution check would leave the exponentiation running on NumPy.
        """
        with patch.object(config, "numpy_available", return_value=False):
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


#: Every public dtype. Integer and floating operands are exercised
#: separately, since section 12.5 gives them different rules.
INTEGER_DTYPES = ("int8", "uint8", "int16", "int32", "int64")
FLOAT_DTYPES = ("float32", "float64")


class ExecutionLocationHarness(unittest.TestCase):
    """Runs a case while watching both the provider and the Python kernel."""

    def provider(self, selection):
        import importlib

        return importlib.import_module(f"tensors.backend.{selection}.kernels")

    def assertRunsOnProvider(self, selection, build, *, calls=1, context=""):
        """The provider's power kernel ran, and the Python one did not.

        Storage residency is checked too, but it is not the evidence: a
        result computed in Python and then moved to the device would still
        be device-resident. The call counts are what establish the location.
        """
        provider = self.provider(selection)
        reference = python_reference()
        with (
            patch.object(provider, "power", wraps=provider.power) as accelerated,
            patch.object(reference, "power", wraps=reference.power) as fallback,
        ):
            backend_state._clear_backend_kernel_cache()
            with ts.use_backend(selection):
                result = build()
            backend_state._clear_backend_kernel_cache()

        self.assertEqual(
            fallback.call_count,
            0,
            f"{context}: the Python power kernel ran under {selection} selection",
        )
        self.assertEqual(
            accelerated.call_count,
            calls,
            f"{context}: expected {calls} {selection} power calls, "
            f"got {accelerated.call_count}",
        )
        if result is not None and hasattr(result, "_storage"):
            self.assertEqual(storage_name(result), STORAGE_FOR[selection], context)
        return result


class EveryDtypeRunsOnTheSelectedBackend(ExecutionLocationHarness):
    """All seven public dtypes, all three operand forms, several sizes."""

    def selections(self):
        return [name for name in ("numpy", "cuda") if name in ts.available_backends()]

    def test_integer_dtypes(self):
        for dtype_name in INTEGER_DTYPES:
            dtype = getattr(ts, dtype_name)
            for size in (1, 4, 64, 1000):
                for selection in self.selections():
                    context = f"{selection}/{dtype_name}/size {size}"
                    with self.subTest(dtype=dtype_name, size=size, backend=selection):
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: ts.Tensor([2] * n, dtype=d)
                            ** ts.Tensor([3] * n, dtype=d),
                            context=context + " tensor ** tensor",
                        )
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: ts.Tensor([2] * n, dtype=d) ** 3,
                            context=context + " tensor ** scalar",
                        )
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: 2 ** ts.Tensor([3] * n, dtype=d),
                            context=context + " scalar ** tensor",
                        )

    def test_floating_dtypes(self):
        for dtype_name in FLOAT_DTYPES:
            dtype = getattr(ts, dtype_name)
            for size in (1, 4, 64, 1000):
                for selection in self.selections():
                    context = f"{selection}/{dtype_name}/size {size}"
                    with self.subTest(dtype=dtype_name, size=size, backend=selection):
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: ts.Tensor([2.0] * n, dtype=d)
                            ** ts.Tensor([3.0] * n, dtype=d),
                            context=context + " tensor ** tensor",
                        )
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: ts.Tensor([2.0] * n, dtype=d)
                            ** 3.0,
                            context=context + " tensor ** scalar",
                        )
                        self.assertRunsOnProvider(
                            selection,
                            lambda d=dtype, n=size: 2.0
                            ** ts.Tensor([3.0] * n, dtype=d),
                            context=context + " scalar ** tensor",
                        )

    def test_integer_wraparound_runs_on_the_provider(self):
        """Section 12.4.1's wraparound is a result, not a reason to decline."""
        for dtype_name in INTEGER_DTYPES:
            dtype = getattr(ts, dtype_name)
            for selection in self.selections():
                with self.subTest(dtype=dtype_name, backend=selection):
                    produced = self.assertRunsOnProvider(
                        selection,
                        lambda d=dtype: ts.Tensor([7] * 64, dtype=d) ** 40,
                        context=f"{selection}/{dtype_name} wraparound",
                    )
                    with ts.use_backend("python"):
                        expected = (ts.Tensor([7] * 64, dtype=dtype) ** 40).tolist()
                    self.assertEqual(produced.tolist(), expected)

    def test_floating_special_values_run_on_the_provider(self):
        """Section 12.3.3's exceptional rows are values, not declines."""
        import math

        cases = (
            ("negative base, fractional exponent", -2.0, 0.5),
            ("zero base, negative exponent", 0.0, -1.0),
            ("negative zero base", -0.0, 3.0),
            ("overflow", 1e200, 2.0),
            ("infinite base", math.inf, 2.0),
            ("nan base", math.nan, 2.0),
            ("zero exponent", 5.0, 0.0),
        )
        for label, base, exponent in cases:
            for dtype_name in FLOAT_DTYPES:
                dtype = getattr(ts, dtype_name)
                for selection in self.selections():
                    with self.subTest(case=label, dtype=dtype_name, backend=selection):
                        self.assertRunsOnProvider(
                            selection,
                            lambda b=base, e=exponent, d=dtype: ts.Tensor(
                                [b] * 64, dtype=d
                            )
                            ** e,
                            context=f"{selection}/{dtype_name} {label}",
                        )

    def test_subnormal_operands_run_on_the_provider(self):
        smallest = {"float32": 1.401298464324817e-45, "float64": 5e-324}
        for dtype_name in FLOAT_DTYPES:
            dtype = getattr(ts, dtype_name)
            for selection in self.selections():
                with self.subTest(dtype=dtype_name, backend=selection):
                    self.assertRunsOnProvider(
                        selection,
                        lambda d=dtype, v=smallest[dtype_name]: ts.Tensor(
                            [v] * 64, dtype=d
                        )
                        ** 1.0,
                        context=f"{selection}/{dtype_name} subnormal",
                    )

    def test_mixed_dtype_promotion_runs_on_the_provider(self):
        for selection in self.selections():
            with self.subTest(backend=selection):
                produced = self.assertRunsOnProvider(
                    selection,
                    lambda: ts.Tensor([2.0] * 64, dtype=ts.float32)
                    ** ts.Tensor([3.0] * 64, dtype=ts.float64),
                    context=f"{selection} float32 ** float64",
                )
                self.assertIs(produced.dtype, ts.float64)

    def test_broadcasting_runs_on_the_provider(self):
        for selection in self.selections():
            with self.subTest(backend=selection):
                produced = self.assertRunsOnProvider(
                    selection,
                    lambda: ts.Tensor([[2.0], [3.0], [4.0]], dtype=ts.float64)
                    ** ts.Tensor([[2.0, 3.0]], dtype=ts.float64),
                    context=f"{selection} broadcast",
                )
                self.assertEqual(produced.shape, (3, 2))

    def test_a_non_contiguous_operand_runs_on_the_provider(self):
        for selection in self.selections():
            with self.subTest(backend=selection):
                self.assertRunsOnProvider(
                    selection,
                    lambda: ts.Tensor([2.0] * 128, dtype=ts.float64)[::2] ** 3.0,
                    context=f"{selection} strided view",
                )

    def test_graph_replay_runs_on_the_provider(self):
        from tensors.graph import Computation

        for selection in self.selections():
            with self.subTest(backend=selection):
                self.assertRunsOnProvider(
                    selection,
                    lambda: Computation(
                        ts.Variable(
                            ts.Tensor([2.0] * 64, dtype=ts.float64),
                            requires_grad=False,
                        )
                        ** ts.Variable(
                            ts.Tensor([3.0] * 64, dtype=ts.float64),
                            requires_grad=False,
                        )
                    ).forward(),
                    # Building the expression evaluates it once; the replay
                    # evaluates it again.
                    calls=2,
                    context=f"{selection} graph replay",
                )

    @requires_numpy
    def test_automatic_selection_uses_its_resolved_backend(self):
        """``auto`` resolves to NumPy here, and follows the same rule."""
        provider = self.provider("numpy")
        reference = python_reference()
        with (
            patch.object(provider, "power", wraps=provider.power) as accelerated,
            patch.object(reference, "power", wraps=reference.power) as fallback,
        ):
            backend_state._clear_backend_kernel_cache()
            with ts.use_backend("auto"):
                ts.full((4,), 2.0, dtype=ts.float64) ** 3.0
            backend_state._clear_backend_kernel_cache()
        self.assertEqual(accelerated.call_count, 1)
        self.assertEqual(fallback.call_count, 0)


@requires_cuda
class CudaStaysOnTheDevice(ExecutionLocationHarness):
    """Residency, and no operand-dependent host read in the dispatcher."""

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
        import math

        cases = (
            ("ordinary", 2.0, 3.0, ts.float64),
            ("float32", 2.0, 3.0, ts.float32),
            ("negative base", -2.0, 0.5, ts.float64),
            ("zero base", 0.0, -1.0, ts.float64),
            ("infinite base", math.inf, 2.0, ts.float64),
            ("integer", 2, 10, ts.int32),
            ("integer wraparound", 7, 40, ts.int64),
        )
        for label, base, exponent, dtype in cases:
            for size in (1, 64, 100_000):
                with self.subTest(case=label, size=size):
                    with ts.use_backend("cuda"):
                        produced = ts.Tensor([base] * size, dtype=dtype) ** exponent
                    self.assertEqual(storage_name(produced), "CudaStorage", label)
                    self.assertIs(produced.dtype, dtype)

    def test_no_operand_is_read_back_to_the_host(self):
        import math

        mixed = [-2.0, 0.0, 2.0, math.inf, math.nan, -0.0]
        for size in (1, 64, 100_000):
            with self.subTest(size=size):
                with ts.use_backend("cuda"):
                    values = (mixed * (size // len(mixed) + 1))[:size]
                    base = ts.Tensor(values, dtype=ts.float64) + 0.0
                    exponent = ts.full((size,), 0.5, dtype=ts.float64) + 0.0
                    with self._counting_device_reads() as reads:
                        produced = base**exponent
                        self.assertEqual(storage_name(produced), "CudaStorage")
                self.assertEqual(reads.count, 0, f"size {size} materialised an operand")

    def test_integer_power_is_native_on_cuda(self):
        """The dispatcher used to document CUDA as unable to do this."""
        for dtype_name in INTEGER_DTYPES:
            dtype = getattr(ts, dtype_name)
            with self.subTest(dtype=dtype_name):
                produced = self.assertRunsOnProvider(
                    "cuda",
                    lambda d=dtype: ts.Tensor([2] * 64, dtype=d) ** 3,
                    context=f"cuda/{dtype_name} integer power",
                )
                self.assertEqual(produced.tolist(), [8] * 64)
                self.assertEqual(storage_name(produced), "CudaStorage")


@requires_numpy
class TheProviderNoLongerDeclines(unittest.TestCase):
    """What used to be power's decline path.

    Power's kernels returned ``None`` for three different things: a domain
    error, CUDA's missing integer exponentiation, and an integer result that
    left the declared dtype. D1 to D4 removed all three — domain violations
    are values (§12.3), integer exponentiation is native and wraps (§12.4.1),
    and CUDA computes integers itself. The kernel is therefore expected to
    answer every supported call.

    The dispatcher no longer routes a ``None`` anywhere: it raises, as the
    four contract operations do.
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
                    from tensors.backend.numpy.conversion import _arithmetic_operand

                    storage = numpy_kernels.power(
                        _arithmetic_operand(base, dtype),
                        _arithmetic_operand(exponent, dtype),
                        dtype=dtype,
                        output_shape=(64,),
                    )
                self.assertIsNotNone(
                    storage, f"{label} still declines to the reference"
                )

    def test_a_forced_numpy_decline_raises_rather_than_falling_back(self):
        self.assertDeclineRaises("numpy", numpy_kernels)

    @requires_cuda
    def test_a_forced_cuda_decline_raises_rather_than_falling_back(self):
        import tensors.backend.cuda.kernels as cuda_kernels

        self.assertDeclineRaises("cuda", cuda_kernels)

    def assertDeclineRaises(self, selection, provider):
        """A provider that returns ``None`` is a capability failure.

        The decline is forced by patching the provider, not by finding an
        input that makes it decline — no such input exists any more, and
        inventing a production switch to simulate one would be worse than
        the gap it tested.
        """
        reference = python_reference()
        with patch.object(provider, "power", return_value=None) as declining:
            with patch.object(reference, "power", wraps=reference.power) as fallback:
                backend_state._clear_backend_kernel_cache()
                with ts.use_backend(selection):
                    with self.assertRaises(
                        config.BackendOperationUnsupportedError
                    ) as raised:
                        ts.full((64,), 2.0, dtype=ts.float64) ** 3.0
                backend_state._clear_backend_kernel_cache()

        declining.assert_called_once()
        message = str(raised.exception)
        self.assertIn(selection, message)
        self.assertIn("power", message)
        self.assertIn("float64", message)
        self.assertEqual(
            fallback.call_count,
            0,
            "the Python power kernel answered a declined call",
        )


if __name__ == "__main__":
    unittest.main()
