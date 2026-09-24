"""Selected-backend execution for selection and comparison operations."""

import inspect
import math
import unittest
from unittest.mock import patch

import tensors as ts
import tensors.backend as backend_state
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state

BACKENDS = ("python", "numpy", "cuda")
SIZES = (1, 31, 32, 33, 4096)

FORWARD_NAMES = (
    "clip",
    "where",
    "maximum",
    "minimum",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
)
GRADIENT_NAMES = (
    "clip_gradient",
    "where_gradient",
    "maximum_gradient",
    "minimum_gradient",
)


def forward_results(size):
    values = ts.Tensor([0.5] * size, dtype=ts.float64)
    other = ts.Tensor([1.0] * size, dtype=ts.float64)
    condition = ts.Tensor([1, 0] * (size // 2) + [1] * (size % 2), dtype=ts.uint8)
    return (
        ts.clip(values, 0.0, 1.0),
        ts.where(condition, values, other),
        ts.maximum(values, other),
        ts.minimum(values, other),
        ts.equal(values, other),
        ts.not_equal(values, other),
        ts.less(values, other),
        ts.less_equal(values, other),
        ts.greater(values, other),
        ts.greater_equal(values, other),
    )


def one_forward(name):
    values = ts.Tensor([0.5], dtype=ts.float64)
    other = ts.Tensor([1.0], dtype=ts.float64)
    condition = ts.Tensor([1], dtype=ts.uint8)
    return {
        "clip": lambda: ts.clip(values, 0.0, 1.0),
        "where": lambda: ts.where(condition, values, other),
        "maximum": lambda: ts.maximum(values, other),
        "minimum": lambda: ts.minimum(values, other),
        "equal": lambda: ts.equal(values, other),
        "not_equal": lambda: ts.not_equal(values, other),
        "less": lambda: ts.less(values, other),
        "less_equal": lambda: ts.less_equal(values, other),
        "greater": lambda: ts.greater(values, other),
        "greater_equal": lambda: ts.greater_equal(values, other),
    }[name]()


class SelectionComparisonExecutionTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        backend_state._clear_backend_kernel_cache()
        reset_graph_state()

    def for_each_backend(self, body):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                if backend not in ts.available_backends():
                    self.skipTest(f"the {backend} backend is not available here")
                reset_graph_state()
                with ts.use_backend(backend):
                    body(backend)

    def test_all_forward_operations_stay_selected_at_every_size(self):
        def body(backend):
            for size in SIZES:
                for result in forward_results(size):
                    self.assertEqual(result.backend_storage.kind, backend)
                    self.assertEqual(result.shape, (size,))

        self.for_each_backend(body)

    def test_scalar_inputs_stay_selected(self):
        def body(backend):
            results = (
                ts.clip(ts.Tensor(0.5, dtype=ts.float64), 0.0, 1.0),
                ts.where(1, 2.0, 3.0),
                ts.maximum(2.0, 3.0),
                ts.minimum(2.0, 3.0),
                ts.equal(2.0, 3.0),
                ts.not_equal(2.0, 3.0),
                ts.less(2.0, 3.0),
                ts.less_equal(2.0, 3.0),
                ts.greater(2.0, 3.0),
                ts.greater_equal(2.0, 3.0),
            )
            for result in results:
                self.assertEqual(result.shape, ())
                self.assertEqual(result.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_dispatchers_prepare_broadcasted_native_operands_for_kernels(self):
        def assert_prepared(arguments, backend, output_shape):
            expected_size = math.prod(output_shape)
            for argument in arguments:
                if backend == "python":
                    self.assertEqual(len(argument), expected_size)
                else:
                    self.assertEqual(tuple(argument.shape), output_shape)

        def body(backend):
            kernels = __import__(
                f"tensors.backend.{backend}.kernels", fromlist=["kernels"]
            )

            cases = (
                (
                    "maximum",
                    lambda: ts.maximum(
                        1.5,
                        ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    ),
                    2,
                ),
                (
                    "minimum",
                    lambda: ts.minimum(
                        ts.Tensor([[1.0], [4.0]]),
                        ts.Tensor([[2.0, 3.0, 4.0]]),
                    ),
                    2,
                ),
                (
                    "where",
                    lambda: ts.where(
                        ts.Tensor([[1], [0]], dtype=ts.uint8),
                        ts.Tensor([[1.0, 2.0, 3.0]]),
                        0.0,
                    ),
                    3,
                ),
                (
                    "greater",
                    lambda: ts.greater(
                        ts.Tensor([[1.0], [4.0]]),
                        ts.Tensor([[2.0, 3.0, 4.0]]),
                    ),
                    2,
                ),
            )
            for name, build, operand_count in cases:
                with self.subTest(backend=backend, operation=name):
                    with patch.object(
                        kernels, name, wraps=getattr(kernels, name)
                    ) as kernel:
                        backend_state._clear_backend_kernel_cache()
                        result = build()
                    self.assertEqual(result.shape, (2, 3))
                    self.assertEqual(result.backend_storage.kind, backend)
                    assert_prepared(
                        kernel.call_args.args[:operand_count], backend, (2, 3)
                    )

            for name, operation in (
                ("maximum_gradient", ts.maximum),
                ("minimum_gradient", ts.minimum),
            ):
                reset_graph_state()
                left = ts.Variable(ts.Tensor([[1.0], [4.0]]))
                right = ts.Tensor([[2.0, 3.0, 4.0]])
                with patch.object(
                    kernels,
                    name,
                    wraps=getattr(kernels, name),
                ) as kernel:
                    backend_state._clear_backend_kernel_cache()
                    gradient = ts.grad(ts.sum(operation(left, right)), left)
                self.assertEqual(gradient.backend_storage.kind, backend)
                assert_prepared(kernel.call_args.args[:3], backend, (2, 3))

            reset_graph_state()
            branch = ts.Variable(ts.Tensor([[1.0, 2.0, 3.0]]))
            condition = ts.Tensor([[1], [0]], dtype=ts.uint8)
            with patch.object(
                kernels,
                "where_gradient",
                wraps=kernels.where_gradient,
            ) as kernel:
                backend_state._clear_backend_kernel_cache()
                gradient = ts.grad(ts.sum(ts.where(condition, branch, 0.0)), branch)
            self.assertEqual(gradient.backend_storage.kind, backend)
            assert_prepared(kernel.call_args.args[:2], backend, (2, 3))

        self.for_each_backend(body)

    def test_first_order_vjps_stay_selected_and_reduce_broadcast_axes(self):
        def body(backend):
            cases = []

            value = ts.Variable(ts.Tensor([-1.0, 0.5, 2.0], dtype=ts.float64))
            cases.append((ts.sum(ts.clip(value, 0.0, 1.0)), value, [0.0, 1.0, 0.0]))

            left = ts.Variable(ts.Tensor([[1.0], [2.0]], dtype=ts.float64))
            condition = ts.Tensor([[1, 0, 1], [0, 1, 0]], dtype=ts.uint8)
            cases.append((ts.sum(ts.where(condition, left, 0.0)), left, [2.0, 1.0]))

            maximum_left = ts.Variable(ts.Tensor([[1.0], [4.0]], dtype=ts.float64))
            cases.append(
                (
                    ts.sum(ts.maximum(maximum_left, [2.0, 3.0, 4.0])),
                    maximum_left,
                    [0.0, 2.5],
                )
            )

            minimum_left = ts.Variable(ts.Tensor([[1.0], [4.0]], dtype=ts.float64))
            cases.append(
                (
                    ts.sum(ts.minimum(minimum_left, [2.0, 3.0, 4.0])),
                    minimum_left,
                    [3.0, 0.5],
                )
            )

            for output, variable, expected in cases:
                gradient = ts.grad(output, variable)
                self.assertEqual(gradient.backend_storage.kind, backend)
                self.assertEqual(gradient.shape, variable.shape)
                self.assertEqual(gradient.tolist(), expected)

            reset_graph_state()
            scalar = ts.Variable(ts.Tensor(2.0, dtype=ts.float64))
            output = ts.where(1, scalar, [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
            gradient = ts.grad(ts.sum(output), scalar)
            self.assertEqual(gradient.backend_storage.kind, backend)
            self.assertEqual(gradient.shape, ())
            self.assertEqual(gradient.tolist(), [6.0])

        self.for_each_backend(body)

    def test_nonfinite_where_branch_does_not_corrupt_first_gradient(self):
        def body(backend):
            for nonfinite in (math.inf, math.nan):
                for condition, selected_left in (([1], True), ([0], False)):
                    reset_graph_state()
                    branch = ts.Variable(ts.Tensor([nonfinite], dtype=ts.float64))
                    output = (
                        ts.where(condition, branch, 0.0)
                        if selected_left
                        else ts.where(condition, 0.0, branch)
                    )
                    gradient = ts.grad(ts.sum(output), branch, create_graph=True)
                    self.assertEqual(gradient.data.backend_storage.kind, backend)
                    self.assertEqual(gradient.data.tolist(), [1.0])

        self.for_each_backend(body)

    def test_vjps_obey_the_selection_at_old_threshold_sizes(self):
        def body(backend):
            for size in SIZES:
                for function in (
                    lambda value: ts.clip(value, -1.0, 1.0),
                    lambda value: ts.where([1] * size, value, 0.0),
                    lambda value: ts.maximum(value, -2.0),
                    lambda value: ts.minimum(value, 2.0),
                ):
                    reset_graph_state()
                    variable = ts.Variable(ts.Tensor([0.5] * size, dtype=ts.float64))
                    gradient = ts.grad(ts.sum(function(variable)), variable)
                    self.assertEqual(gradient.backend_storage.kind, backend)
                    self.assertEqual(gradient.size, size)

        self.for_each_backend(body)

    def test_boundary_tie_nan_and_signed_zero_rules_are_preserved(self):
        def body(backend):
            clipped = ts.clip(
                ts.Tensor([-math.inf, -0.0, 0.5, 1.0, math.inf, math.nan]),
                0.0,
                1.0,
            ).tolist()
            self.assertEqual(clipped[:5], [0.0, -0.0, 0.5, 1.0, 1.0])
            self.assertLess(math.copysign(1.0, clipped[1]), 0.0)
            self.assertTrue(math.isnan(clipped[5]))
            self.assertEqual(ts.clip([-2.0, 2.0], min_value=0.0).tolist(), [0.0, 2.0])
            self.assertEqual(ts.clip([-2.0, 2.0], max_value=0.0).tolist(), [-2.0, 0.0])

            variable = ts.Variable(ts.Tensor([0.0, 0.5, 1.0, math.nan]))
            gradient = ts.grad(ts.sum(ts.clip(variable, 0.0, 1.0)), variable).tolist()
            self.assertEqual(gradient[:3], [0.0, 1.0, 0.0])
            self.assertTrue(math.isnan(gradient[3]))

            for function in (ts.maximum, ts.minimum):
                result = function(
                    [-0.0, 0.0, math.nan, 1.0], [0.0, -0.0, 2.0, math.nan]
                ).tolist()
                self.assertLess(math.copysign(1.0, result[0]), 0.0)
                self.assertGreater(math.copysign(1.0, result[1]), 0.0)
                self.assertTrue(math.isnan(result[2]))
                self.assertTrue(math.isnan(result[3]))

            left = ts.Variable([1.0, 2.0, 4.0])
            right = ts.Variable([2.0, 2.0, 3.0])
            left_gradient = ts.grad(ts.sum(ts.maximum(left, right)), left)
            self.assertEqual(left_gradient.tolist(), [0.0, 0.5, 1.0])

        self.for_each_backend(body)

    def test_where_and_comparison_edge_semantics_are_preserved(self):
        def body(backend):
            selected = ts.where([0.0, -0.0, math.nan, -2.0], [1, 2, 3, 4], [5, 6, 7, 8])
            self.assertEqual(selected.tolist(), [5, 6, 3, 4])

            large = 2**60 + 1
            left = ts.Tensor([large, large, -large], dtype=ts.int64)
            right = ts.Tensor([large, large - 1, -large], dtype=ts.int64)
            self.assertEqual(ts.equal(left, right).tolist(), [1, 0, 1])
            self.assertEqual(ts.greater(left, right).tolist(), [0, 1, 0])
            self.assertIs(ts.equal(left, right).dtype, ts.uint8)

            values = ts.Tensor([math.nan, 0.0, -0.0, 1.0])
            self.assertEqual(ts.equal(values, values).tolist(), [0, 1, 1, 1])
            self.assertEqual(ts.not_equal(values, values).tolist(), [1, 0, 0, 0])
            self.assertEqual(ts.less(values, values).tolist(), [0, 0, 0, 0])
            self.assertEqual(ts.less_equal(values, values).tolist(), [0, 1, 1, 1])

        self.for_each_backend(body)

    def test_float32_subnormals_are_not_flushed_during_selection_or_comparison(self):
        def body(backend):
            smallest = 1.401298464324817e-45
            value = ts.Tensor([smallest], dtype=ts.float32)
            self.assertEqual(ts.maximum(value, 0.0).tolist(), [smallest])
            self.assertEqual(ts.minimum(value, 0.0).tolist(), [0.0])
            self.assertEqual(ts.clip(value, 0.0, 1.0).tolist(), [smallest])
            self.assertEqual(ts.greater(value, 0.0).tolist(), [1])
            self.assertEqual(ts.equal(value, 0.0).tolist(), [0])

        self.for_each_backend(body)

    def test_accelerated_selection_never_calls_the_python_kernel(self):
        import tensors.backend.python.kernels as python_kernels

        for backend in ("numpy", "cuda"):
            if backend not in ts.available_backends():
                continue
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    for name in FORWARD_NAMES:
                        with patch.object(
                            python_kernels,
                            name,
                            side_effect=AssertionError("Python fallback ran"),
                        ):
                            backend_state._clear_backend_kernel_cache()
                            result = one_forward(name)
                            self.assertEqual(result.backend_storage.kind, backend)

    def test_a_declining_provider_raises_instead_of_falling_back(self):
        for backend in ("numpy", "cuda"):
            if backend not in ts.available_backends():
                continue
            kernels = __import__(
                f"tensors.backend.{backend}.kernels", fromlist=["kernels"]
            )
            with ts.use_backend(backend):
                for name in FORWARD_NAMES:
                    with self.subTest(backend=backend, operation=name):
                        with patch.object(kernels, name, return_value=None):
                            backend_state._clear_backend_kernel_cache()
                            with self.assertRaises(BackendOperationUnsupportedError):
                                one_forward(name)

    def test_declining_vjp_providers_raise_instead_of_falling_back(self):
        def build(function):
            variable = ts.Variable([0.5])
            return function(variable), variable

        builders = {
            "clip_gradient": lambda: build(
                lambda variable: ts.clip(variable, 0.0, 1.0)
            ),
            "where_gradient": lambda: build(
                lambda variable: ts.where([1], variable, 0.0)
            ),
            "maximum_gradient": lambda: build(
                lambda variable: ts.maximum(variable, 0.0)
            ),
            "minimum_gradient": lambda: build(
                lambda variable: ts.minimum(variable, 1.0)
            ),
        }
        for backend in ("numpy", "cuda"):
            if backend not in ts.available_backends():
                continue
            kernels = __import__(
                f"tensors.backend.{backend}.kernels", fromlist=["kernels"]
            )
            with ts.use_backend(backend):
                for name in GRADIENT_NAMES:
                    with self.subTest(backend=backend, operation=name):
                        reset_graph_state()
                        output, variable = builders[name]()
                        with patch.object(kernels, name, return_value=None):
                            backend_state._clear_backend_kernel_cache()
                            with self.assertRaises(BackendOperationUnsupportedError):
                                ts.grad(ts.sum(output), variable)

    def test_dispatchers_have_no_workload_policy_or_reference_fallback(self):
        from tensors.backend.dispatch.elementwise import (
            clip,
            clip_gradient,
            equal,
            greater,
            greater_equal,
            less,
            less_equal,
            maximum,
            maximum_gradient,
            minimum,
            minimum_gradient,
            not_equal,
            where,
            where_gradient,
        )

        for module in (
            clip,
            clip_gradient,
            where,
            where_gradient,
            maximum,
            maximum_gradient,
            minimum,
            minimum_gradient,
            equal,
            not_equal,
            less,
            less_equal,
            greater,
            greater_equal,
        ):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("_array_work_is_large_enough", source)
                self.assertNotIn("from tensors.backend.policy", source)
                self.assertNotIn("kernels.elementwise", source)
                self.assertIn("validate_backend_residency", source)
                self.assertIn("BackendOperationUnsupportedError", source)

    def test_kernels_contain_no_broadcasting_logic(self):
        import importlib

        names = (
            "where",
            "where_gradient",
            "maximum",
            "maximum_gradient",
            "minimum",
            "minimum_gradient",
            "equal",
            "not_equal",
            "less",
            "less_equal",
            "greater",
            "greater_equal",
        )
        for backend in BACKENDS:
            for name in names:
                with self.subTest(backend=backend, operation=name):
                    module = importlib.import_module(
                        f"tensors.backend.{backend}.kernels.elementwise.{name}"
                    )
                    source = inspect.getsource(module)
                    self.assertNotIn("broadcast_to", source)
                    self.assertNotIn("broadcast_arrays", source)
                    self.assertNotIn("broadcast_tensors", source)

    def test_selection_nodes_remain_ordinary_boundaries_between_fused_runs(self):
        from tensors.graph.computation.fusion import _KERNEL_NAMES

        self.assertTrue(
            {"Clip", "Where", "Maximum", "Minimum"}.isdisjoint(_KERNEL_NAMES)
        )
        if "cuda" not in ts.available_backends():
            self.skipTest("CUDA is not available")
        with ts.use_backend("cuda"):
            variable = ts.Variable(ts.Tensor([0.5] * 4096), requires_grad=False)
            output = ts.maximum(variable * 2.0, 0.25) + 1.0
            result = ts.graph.Computation(output).forward()
        self.assertEqual(result.backend_storage.kind, "cuda")
        self.assertEqual(result.tolist()[:2], [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
