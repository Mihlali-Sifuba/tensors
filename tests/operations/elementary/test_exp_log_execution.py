"""Exp and log under the selected-backend execution contract.

The expected values come from their mathematical definitions and the written
contracts in ``docs/exp-semantics.md`` and ``docs/log-semantics.md``. Another
backend is never used as the oracle.
"""

import importlib
import inspect
import math
import struct
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state

BACKENDS = ("python", "numpy", "cuda")
SIZES = (1, 2, 31, 32, 33, 4096)
SMALLEST_FLOAT32_SUBNORMAL = struct.unpack("!f", struct.pack("!I", 1))[0]


def float32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


class ExpLogTestCase(unittest.TestCase):
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

    @staticmethod
    def gradient(function, values, upstream=None, dtype=ts.float64):
        variable = ts.Variable(ts.Tensor(values, dtype=dtype), requires_grad=True)
        output = function(variable)
        if upstream is None:
            upstream = [1.0] * len(values)
        return ts.grad(
            output,
            variable,
            grad_outputs=ts.Tensor(upstream, dtype=dtype),
        )


class SelectedBackendExecutionTests(ExpLogTestCase):
    def test_forward_executes_on_the_selection_at_every_size(self):
        def body(backend):
            for size in SIZES:
                for function, value in ((ts.exp, 0.5), (ts.log, 2.0)):
                    produced = function(ts.Tensor([value] * size, dtype=ts.float64))
                    self.assertEqual(produced.backend_storage.kind, backend)
                    self.assertEqual(produced.size, size)

        self.for_each_backend(body)

    def test_first_vjp_executes_on_the_selection_at_every_size(self):
        def body(backend):
            for size in SIZES:
                for function, value in ((ts.exp, 0.5), (ts.log, 2.0)):
                    reset_graph_state()
                    produced = self.gradient(function, [value] * size)
                    self.assertEqual(produced.backend_storage.kind, backend)
                    self.assertEqual(produced.size, size)

        self.for_each_backend(body)

    def test_dispatchers_have_no_threshold_or_fallback(self):
        modules = (
            "tensors.backend.dispatch.elementwise.exp",
            "tensors.backend.dispatch.elementwise.exp_gradient",
            "tensors.backend.dispatch.elementwise.log",
            "tensors.backend.dispatch.elementwise.log_gradient",
        )
        forbidden = (
            "_array_work_is_large_enough",
            "_NUMPY_ELEMENTWISE_MIN_SIZE",
            "_backend_kernel",
            "python.kernels",
            "as reference",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                source = inspect.getsource(importlib.import_module(module_name))
                for name in forbidden:
                    self.assertNotIn(name, source)
                self.assertIn("validate_backend_residency", source)
                self.assertIn("load_backend", source)
                self.assertIn("BackendOperationUnsupportedError", source)

    def test_backend_decline_is_an_error_not_a_fallback(self):
        from tensors.backend import dispatch as backend_dispatch
        from tensors.backend.loading import _clear_backend_kernel_cache

        def body(backend):
            if backend == "python":
                return
            kernels = importlib.import_module(f"tensors.backend.{backend}.kernels")
            cases = (
                ("exp", backend_dispatch.execute_exp, (ts.Tensor([0.5]),)),
                ("log", backend_dispatch.execute_log, (ts.Tensor([2.0]),)),
                (
                    "exp_gradient",
                    backend_dispatch.execute_exp_gradient,
                    (ts.Tensor([1.0]), ts.Tensor([0.5])),
                ),
                (
                    "log_gradient",
                    backend_dispatch.execute_log_gradient,
                    (ts.Tensor([1.0]), ts.Tensor([2.0])),
                ),
            )
            for name, execute, operands in cases:
                with self.subTest(backend=backend, operation=name):
                    with patch.object(kernels, name, return_value=None):
                        _clear_backend_kernel_cache()
                        with self.assertRaises(BackendOperationUnsupportedError):
                            execute(
                                *operands,
                                dtype=ts.float64,
                                output_shape=(1,),
                            )
            _clear_backend_kernel_cache()

        self.for_each_backend(body)

    def test_numpy_kernels_receive_native_arrays(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not installed")
        import numpy
        import tensors.backend.numpy.kernels as kernels

        for name, function, value in (("exp", ts.exp, 0.5), ("log", ts.log, 2.0)):
            seen = {}
            original = getattr(kernels, name)

            def spy(values, **keywords):
                seen["values"] = values
                return original(values, **keywords)

            with self.subTest(operation=name), patch.object(kernels, name, spy):
                with ts.use_backend("numpy"):
                    function(ts.Tensor([[value, value]], dtype=ts.float32))
            self.assertIsInstance(seen["values"], numpy.ndarray)
            self.assertEqual(seen["values"].shape, (1, 2))
            self.assertEqual(seen["values"].dtype, numpy.dtype("float32"))


class ExpNumericalContractTests(ExpLogTestCase):
    def test_ordinary_values_and_signed_zero(self):
        def body(backend):
            values = [-2.0, -0.0, 0.0, 1.0, 2.0]
            produced = ts.exp(ts.Tensor(values, dtype=ts.float64)).tolist()
            for got, value in zip(produced, values):
                self.assertAlmostEqual(got, math.exp(value), places=14)
            self.assertEqual(produced[1:3], [1.0, 1.0])

        self.for_each_backend(body)

    def test_infinity_nan_overflow_and_underflow(self):
        def body(backend):
            produced = ts.exp(
                ts.Tensor([math.inf, -math.inf, math.nan, 1000.0, -1000.0])
            ).tolist()
            self.assertEqual(produced[0], math.inf)
            self.assertEqual(produced[1], 0.0)
            self.assertTrue(math.isnan(produced[2]))
            self.assertEqual(produced[3], math.inf)
            self.assertEqual(produced[4], 0.0)

        self.for_each_backend(body)

    def test_float32_overflow_and_gradual_underflow(self):
        def body(backend):
            produced = ts.exp(ts.Tensor([100.0, -90.0], dtype=ts.float32))
            values = produced.tolist()
            self.assertIs(produced.dtype, ts.float32)
            self.assertEqual(values[0], math.inf)
            self.assertEqual(values[1], float32(math.exp(-90.0)))
            self.assertGreater(values[1], 0.0)
            self.assertLess(values[1], 1.1754943508222875e-38)

        self.for_each_backend(body)

    def test_first_vjp_uses_upstream_times_exp(self):
        def body(backend):
            values = [-2.0, 0.0, 2.0]
            upstream = [3.0, -2.0, 0.5]
            produced = self.gradient(ts.exp, values, upstream).tolist()
            for got, gradient, value in zip(produced, upstream, values):
                self.assertAlmostEqual(got, gradient * math.exp(value), places=13)

        self.for_each_backend(body)

    def test_first_vjp_preserves_ieee_exceptional_products(self):
        def body(backend):
            produced = self.gradient(
                ts.exp,
                [math.inf, -math.inf, math.nan, math.inf],
                [1.0, 1.0, 1.0, 0.0],
            ).tolist()
            self.assertEqual(produced[0], math.inf)
            self.assertEqual(produced[1], 0.0)
            self.assertTrue(math.isnan(produced[2]))
            self.assertTrue(math.isnan(produced[3]))

        self.for_each_backend(body)


class LogNumericalContractTests(ExpLogTestCase):
    def test_ordinary_values_infinity_and_nan(self):
        def body(backend):
            values = [0.5, 1.0, math.e, 2.0, math.inf, math.nan]
            produced = ts.log(ts.Tensor(values, dtype=ts.float64)).tolist()
            for got, value in zip(produced[:4], values[:4]):
                self.assertAlmostEqual(got, math.log(value), places=14)
            self.assertEqual(produced[4], math.inf)
            self.assertTrue(math.isnan(produced[5]))

        self.for_each_backend(body)

    def test_non_positive_values_raise_including_signed_zero(self):
        def body(backend):
            for value in (0.0, -0.0, -1.0, -math.inf):
                with self.subTest(backend=backend, value=value):
                    with self.assertRaisesRegex(ValueError, "positive"):
                        ts.log(ts.Tensor([value], dtype=ts.float64))

        self.for_each_backend(body)

    def test_positive_float32_subnormal_is_not_confused_with_zero(self):
        expected = float32(math.log(SMALLEST_FLOAT32_SUBNORMAL))

        def body(backend):
            produced = ts.log(ts.Tensor([SMALLEST_FLOAT32_SUBNORMAL], dtype=ts.float32))
            self.assertIs(produced.dtype, ts.float32)
            self.assertEqual(produced.tolist(), [expected])

        self.for_each_backend(body)

    def test_first_vjp_is_one_division(self):
        def body(backend):
            values = [0.5, 1.0, 2.0, math.inf]
            upstream = [3.0, -2.0, 0.5, 1.0]
            produced = self.gradient(ts.log, values, upstream).tolist()
            expected = [g / x for g, x in zip(upstream, values)]
            self.assertEqual(produced, expected)

        self.for_each_backend(body)

    def test_nan_primal_produces_nan_forward_and_gradient(self):
        def body(backend):
            produced = self.gradient(ts.log, [math.nan], [1.0]).tolist()[0]
            self.assertTrue(math.isnan(produced))

        self.for_each_backend(body)


class ShapeDtypeAndDemandTests(ExpLogTestCase):
    def test_shape_and_floating_dtype_are_preserved(self):
        def body(backend):
            for dtype in (ts.float32, ts.float64):
                for function, values in (
                    (ts.exp, [[0.0, 1.0], [2.0, 3.0]]),
                    (ts.log, [[1.0, 2.0], [3.0, 4.0]]),
                ):
                    produced = function(ts.Tensor(values, dtype=dtype))
                    self.assertEqual(tuple(produced.shape), (2, 2))
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_every_integer_dtype_promotes_to_float64(self):
        integer_dtypes = (ts.int64, ts.int32, ts.int16, ts.int8, ts.uint8)

        def body(backend):
            for dtype in integer_dtypes:
                for function, values in ((ts.exp, [0, 1]), (ts.log, [1, 2])):
                    produced = function(ts.Tensor(values, dtype=dtype))
                    self.assertIs(produced.dtype, ts.float64)
                    self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_noncontiguous_and_empty_operands_preserve_logical_shape(self):
        def body(backend):
            exp_source = ts.Tensor([0.0, 9.0, 1.0, 9.0], dtype=ts.float64)[::2]
            log_source = ts.Tensor([1.0, 9.0, 2.0, 9.0], dtype=ts.float64)[::2]
            self.assertEqual(ts.exp(exp_source).shape, exp_source.shape)
            self.assertEqual(ts.log(log_source).shape, log_source.shape)
            for function in (ts.exp, ts.log):
                empty = function(ts.Tensor([], dtype=ts.float64, shape=(0,)))
                self.assertEqual(empty.tolist(), [])
                self.assertEqual(empty.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_malformed_upstream_gradients_are_rejected_by_the_operation(self):
        from tensors.operations.elementary.exp import Exp
        from tensors.operations.elementary.log import Log

        for operation in (Exp(), Log()):
            with self.subTest(operation=operation.name):
                with self.assertRaisesRegex(ValueError, "does not match value shape"):
                    operation.backward(
                        ts.Tensor([1.0, 1.0], dtype=ts.float64),
                        ts.Tensor([1.0], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )
                with self.assertRaisesRegex(ValueError, "does not match value dtype"):
                    operation.backward(
                        ts.Tensor([1.0], dtype=ts.float32),
                        ts.Tensor([1.0], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )

    def test_unrequested_gradient_returns_none_without_dispatch(self):
        from tensors.backend import dispatch as backend_dispatch
        from tensors.operations.elementary.exp import Exp
        from tensors.operations.elementary.log import Log

        cases = (
            (Exp(), "execute_exp_gradient"),
            (Log(), "execute_log_gradient"),
        )
        for operation, dispatcher in cases:
            with (
                self.subTest(operation=operation.name),
                patch.object(
                    backend_dispatch,
                    dispatcher,
                    side_effect=AssertionError("unrequested VJP must not execute"),
                ),
            ):
                self.assertEqual(
                    operation.backward(
                        ts.Tensor([1.0, 2.0], dtype=ts.float32),
                        ts.Tensor([1.0], dtype=ts.float64),
                        needs_input_grad=(False,),
                    ),
                    [None],
                )


class KernelBoundaryTests(unittest.TestCase):
    def test_migrated_kernels_have_no_registry_wrappers_or_tensor_lowering(self):
        modules = tuple(
            f"tensors.backend.{backend}.kernels.elementwise.{name}"
            for backend in BACKENDS
            for name in ("exp", "exp_gradient", "log", "log_gradient")
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                source = inspect.getsource(importlib.import_module(module_name))
                for forbidden in (
                    "functions =",
                    "evaluate =",
                    "_working_values",
                    "tensor_to_logical_array",
                    "def _exp(",
                    "def _log(",
                ):
                    self.assertNotIn(forbidden, source)


@unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is not available")
class CudaFusionContractTests(ExpLogTestCase):
    def fused(self, function, values):
        import tensors.backend.cuda.kernels as kernels

        with ts.use_backend("cuda"):
            variable = ts.Variable(
                ts.Tensor(values, dtype=ts.float32), requires_grad=False
            )
            output = function(variable) + 0.0
            computation = ts.graph.Computation(output)
            variable.data = ts.Tensor(values, dtype=ts.float32)
            state = {}
            original = kernels.fused_elementwise

            def spy(*arguments, **keywords):
                result = original(*arguments, **keywords)
                state["called"] = True
                state["declined"] = result is None
                return result

            with patch.object(kernels, "fused_elementwise", spy):
                produced = computation.forward()
        self.assertTrue(state.get("called"))
        self.assertFalse(state["declined"])
        self.assertEqual(produced.backend_storage.kind, "cuda")
        return produced.tolist()

    def test_fused_exp_matches_the_written_contract(self):
        probe = [-90.0, -0.0, 0.0, 1.0, 100.0, math.inf, -math.inf, math.nan]
        values = (probe * 600)[:4096]
        produced = self.fused(ts.exp, values)[: len(probe)]
        expected = [float32(math.exp(x)) if x < 89.0 else math.inf for x in probe[:-1]]
        expected.append(math.nan)
        for got, want in zip(produced, expected):
            if math.isnan(want):
                self.assertTrue(math.isnan(got))
            else:
                self.assertEqual(got, want)

    def test_fused_log_matches_the_written_contract(self):
        probe = [SMALLEST_FLOAT32_SUBNORMAL, 0.5, 1.0, 2.0, math.inf, math.nan]
        values = (probe * 700)[:4096]
        produced = self.fused(ts.log, values)[: len(probe)]
        expected = [float32(math.log(x)) for x in probe[:-1]] + [math.nan]
        for got, want in zip(produced, expected):
            if math.isnan(want):
                self.assertTrue(math.isnan(got))
            else:
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
