"""Ordinary trig operations under strict selected-backend execution."""

import importlib
import inspect
import math
import unittest
from unittest.mock import patch

import tensors as ts
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state

BACKENDS = ("python", "numpy", "cuda")
FORWARD_CASES = (
    ("sin", ts.sin, 0.25, math.sin),
    ("cos", ts.cos, 0.25, math.cos),
    ("tan", ts.tan, 0.25, math.tan),
    ("arcsin", ts.arcsin, 0.25, math.asin),
    ("arccos", ts.arccos, 0.25, math.acos),
    ("arctan", ts.arctan, 0.25, math.atan),
)
DERIVATIVES = {
    "sin": lambda x: math.cos(x),
    "cos": lambda x: -math.sin(x),
    "tan": lambda x: 1.0 / math.cos(x) ** 2,
    "arcsin": lambda x: 1.0 / math.sqrt(1.0 - x * x),
    "arccos": lambda x: -1.0 / math.sqrt(1.0 - x * x),
    "arctan": lambda x: 1.0 / (1.0 + x * x),
}


class TrigExecutionTestCase(unittest.TestCase):
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
        if upstream is None:
            upstream = [1.0] * len(values)
        return ts.grad(
            function(variable),
            variable,
            grad_outputs=ts.Tensor(upstream, dtype=dtype),
        )


class SelectedBackendTests(TrigExecutionTestCase):
    def test_forward_runs_on_the_selection_at_every_size(self):
        def body(backend):
            for size in (1, 2, 7, 31, 32, 33, 4096):
                for name, function, value, _ in FORWARD_CASES:
                    with self.subTest(backend=backend, operation=name, size=size):
                        produced = function(ts.Tensor([value] * size, dtype=ts.float64))
                        self.assertEqual(produced.backend_storage.kind, backend)
                        self.assertEqual(produced.size, size)

        self.for_each_backend(body)

    def test_first_vjp_runs_on_the_selection_at_every_size(self):
        def body(backend):
            for size in (1, 2, 31, 32, 4096):
                for name, function, value, _ in FORWARD_CASES:
                    with self.subTest(backend=backend, operation=name, size=size):
                        reset_graph_state()
                        produced = self.gradient(function, [value] * size)
                        self.assertEqual(produced.backend_storage.kind, backend)
                        self.assertEqual(produced.size, size)

        self.for_each_backend(body)

    def test_scalar_and_one_element_inputs_stay_on_the_selection(self):
        def body(backend):
            for name, function, value, _ in FORWARD_CASES:
                with self.subTest(backend=backend, operation=name):
                    scalar = function(ts.Tensor(value, dtype=ts.float64))
                    vector = function(ts.Tensor([value], dtype=ts.float64))
                    self.assertEqual(scalar.shape, ())
                    self.assertEqual(vector.shape, (1,))
                    self.assertEqual(scalar.backend_storage.kind, backend)
                    self.assertEqual(vector.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_accelerated_selection_never_calls_the_python_kernel(self):
        accelerated = [b for b in ts.available_backends() if b in ("numpy", "cuda")]
        for backend in accelerated:
            for name, function, value, _ in FORWARD_CASES:
                module = importlib.import_module(
                    f"tensors.backend.python.kernels.elementwise.{name}"
                )
                with (
                    self.subTest(backend=backend, operation=name),
                    patch.object(
                        module,
                        name,
                        side_effect=AssertionError("unexpected Python fallback"),
                    ),
                ):
                    with ts.use_backend(backend):
                        produced = function(ts.Tensor([value], dtype=ts.float64))
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_dispatchers_contain_no_workload_policy_or_fallback(self):
        for name, *_ in FORWARD_CASES:
            for suffix in ("", "_gradient"):
                module_name = f"tensors.backend.dispatch.elementwise.{name}{suffix}"
                source = inspect.getsource(importlib.import_module(module_name))
                with self.subTest(module=module_name):
                    for forbidden in (
                        "_NUMPY_ELEMENTWISE_MIN_SIZE",
                        "_array_work_is_large_enough",
                        "_backend_kernel",
                        "python.kernels",
                        "as reference",
                    ):
                        self.assertNotIn(forbidden, source)
                    self.assertIn("validate_backend_residency", source)
                    self.assertIn("BackendOperationUnsupportedError", source)

    def test_a_declining_selected_kernel_raises(self):
        from tensors.backend import dispatch as dispatch
        from tensors.backend.loading import _clear_backend_kernel_cache

        for backend in (b for b in ts.available_backends() if b in ("numpy", "cuda")):
            kernels = importlib.import_module(f"tensors.backend.{backend}.kernels")
            with ts.use_backend(backend):
                for name, _, value, _ in FORWARD_CASES:
                    tensor = ts.Tensor([value], dtype=ts.float64)
                    execute = getattr(dispatch, f"execute_{name}")
                    with (
                        self.subTest(backend=backend, operation=name),
                        patch.object(kernels, name, return_value=None),
                    ):
                        _clear_backend_kernel_cache()
                        with self.assertRaises(BackendOperationUnsupportedError):
                            execute(tensor, dtype=ts.float64, output_shape=tensor.shape)
                    execute_gradient = getattr(dispatch, f"execute_{name}_gradient")
                    with (
                        self.subTest(backend=backend, operation=f"{name}_gradient"),
                        patch.object(kernels, f"{name}_gradient", return_value=None),
                    ):
                        _clear_backend_kernel_cache()
                        with self.assertRaises(BackendOperationUnsupportedError):
                            execute_gradient(
                                tensor,
                                tensor,
                                dtype=ts.float64,
                                output_shape=tensor.shape,
                            )
            _clear_backend_kernel_cache()


class NumericalContractTests(TrigExecutionTestCase):
    def test_ordinary_forward_values_and_first_vjps(self):
        values = [-0.75, -0.25, 0.0, 0.25, 0.75]
        upstream = [2.0, -1.0, 0.5, 3.0, -2.0]

        def body(backend):
            for name, function, _, reference in FORWARD_CASES:
                produced = function(ts.Tensor(values, dtype=ts.float64)).tolist()
                gradient = self.gradient(function, values, upstream).tolist()
                for got, x in zip(produced, values):
                    self.assertAlmostEqual(got, reference(x), places=13)
                for got, g, x in zip(gradient, upstream, values):
                    self.assertAlmostEqual(got, g * DERIVATIVES[name](x), places=12)

        self.for_each_backend(body)

    def test_nan_propagates_through_forward_and_vjp(self):
        def body(backend):
            for name, function, _, _ in FORWARD_CASES:
                with self.subTest(backend=backend, operation=name):
                    self.assertTrue(
                        math.isnan(function(ts.Tensor([math.nan])).tolist()[0])
                    )
                    self.assertTrue(
                        math.isnan(self.gradient(function, [math.nan]).tolist()[0])
                    )

        self.for_each_backend(body)

    def test_direct_trig_functions_reject_infinity(self):
        def body(backend):
            for function in (ts.sin, ts.cos, ts.tan):
                for value in (math.inf, -math.inf):
                    with self.assertRaises(ValueError):
                        function(ts.Tensor([value], dtype=ts.float64))

        self.for_each_backend(body)

    def test_inverse_domains_include_endpoints_and_exclude_neighbours(self):
        inside = math.nextafter(1.0, 0.0)
        outside = math.nextafter(1.0, math.inf)

        def body(backend):
            for function in (ts.arcsin, ts.arccos):
                produced = function(
                    ts.Tensor([-1.0, -inside, inside, 1.0], dtype=ts.float64)
                )
                self.assertTrue(all(math.isfinite(x) for x in produced.tolist()))
                for value in (-outside, outside, -math.inf, math.inf):
                    with self.assertRaisesRegex(ValueError, "between -1 and 1"):
                        function(ts.Tensor([value], dtype=ts.float64))

        self.for_each_backend(body)

    def test_inverse_endpoint_vjps_remain_undefined(self):
        def body(backend):
            for function in (ts.arcsin, ts.arccos):
                for value in (-1.0, 1.0):
                    reset_graph_state()
                    variable = ts.Variable(ts.Tensor([value], dtype=ts.float64))
                    with self.assertRaisesRegex(ValueError, "undefined at -1 and 1"):
                        ts.grad(function(variable), variable)

        self.for_each_backend(body)

    def test_signed_zero_and_arctan_infinities(self):
        def body(backend):
            for function in (ts.sin, ts.tan, ts.arcsin, ts.arctan):
                produced = function(ts.Tensor([0.0, -0.0], dtype=ts.float64)).tolist()
                self.assertEqual(math.copysign(1.0, produced[0]), 1.0)
                self.assertEqual(math.copysign(1.0, produced[1]), -1.0)
            produced = ts.arctan(ts.Tensor([math.inf, -math.inf])).tolist()
            self.assertEqual(produced, [math.pi / 2.0, -math.pi / 2.0])
            gradient = self.gradient(ts.arctan, [1.0e308, math.inf, -math.inf])
            self.assertEqual(gradient.tolist(), [0.0, 0.0, 0.0])

        self.for_each_backend(body)


class TensorSemanticsTests(TrigExecutionTestCase):
    def test_float_dtype_and_shape_are_preserved(self):
        def body(backend):
            for dtype in (ts.float32, ts.float64):
                for name, function, value, _ in FORWARD_CASES:
                    produced = function(
                        ts.Tensor([[value, value], [value, value]], dtype=dtype)
                    )
                    self.assertEqual(tuple(produced.shape), (2, 2))
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_every_integer_dtype_promotes_to_float64(self):
        dtypes = (ts.int64, ts.int32, ts.int16, ts.int8, ts.uint8)

        def body(backend):
            for dtype in dtypes:
                for name, function, _, _ in FORWARD_CASES:
                    with self.subTest(
                        backend=backend, dtype=dtype.name, operation=name
                    ):
                        produced = function(ts.Tensor([0], dtype=dtype))
                        self.assertIs(produced.dtype, ts.float64)
                        self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_backward_validates_shape_and_dtype_and_respects_demand(self):
        classes = (
            ("sin", "Sin"),
            ("cos", "Cos"),
            ("tan", "Tan"),
            ("arcsin", "ArcSin"),
            ("arccos", "ArcCos"),
            ("arctan", "ArcTan"),
        )
        for module_name, class_name in classes:
            operation = getattr(
                importlib.import_module(
                    f"tensors.operations.trigonometric.{module_name}"
                ),
                class_name,
            )()
            with self.subTest(operation=module_name):
                with self.assertRaisesRegex(ValueError, "does not match value shape"):
                    operation.backward(
                        ts.Tensor([1.0, 1.0], dtype=ts.float64),
                        ts.Tensor([0.25], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )
                with self.assertRaisesRegex(ValueError, "does not match value dtype"):
                    operation.backward(
                        ts.Tensor([1.0], dtype=ts.float32),
                        ts.Tensor([0.25], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )
                self.assertEqual(
                    operation.backward(
                        ts.Tensor([1.0, 1.0], dtype=ts.float32),
                        ts.Tensor([0.25], dtype=ts.float64),
                        needs_input_grad=(False,),
                    ),
                    [None],
                )

    def test_numpy_kernels_receive_native_values(self):
        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is unavailable")
        import numpy
        import tensors.backend.numpy.kernels as kernels

        for name, function, value, _ in FORWARD_CASES:
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


@unittest.skipUnless("cuda" in ts.available_backends(), "CUDA is unavailable")
class CudaFusionCompatibilityTests(TrigExecutionTestCase):
    def test_fused_forward_and_vjp_match_the_mathematical_rules(self):
        import tensors.backend.cuda.kernels as kernels
        import tensors.backend as backend_state

        for name, function, value, reference in FORWARD_CASES:
            with self.subTest(operation=name), ts.use_backend("cuda"):
                reset_graph_state()
                variable = ts.Variable(ts.Tensor([value] * 4096, dtype=ts.float64))
                output = function(variable) + 1.0
                computation = ts.graph.Computation(output)
                with patch.object(
                    kernels,
                    "fused_elementwise",
                    wraps=kernels.fused_elementwise,
                ) as forward_fusion:
                    backend_state._clear_backend_kernel_cache()
                    produced = computation.forward()
                with patch.object(
                    kernels,
                    "fused_elementwise_backward",
                    wraps=kernels.fused_elementwise_backward,
                ) as backward_fusion:
                    backend_state._clear_backend_kernel_cache()
                    computation.backward(ts.ones((4096,), dtype=ts.float64))
                forward_fusion.assert_called_once()
                backward_fusion.assert_called_once()
                self.assertAlmostEqual(
                    produced.tolist()[0], reference(value) + 1.0, places=12
                )
                self.assertAlmostEqual(
                    variable.grad.tolist()[0], DERIVATIVES[name](value), places=12
                )
                self.assertEqual(produced.backend_storage.kind, "cuda")
                self.assertEqual(variable.grad.backend_storage.kind, "cuda")


if __name__ == "__main__":
    unittest.main()
