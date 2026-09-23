"""Sigmoid and softplus under the selected-backend execution contract.

`docs/backends.md`, *Execution requirements*: an explicit selection is an
execution requirement, not a performance preference. Both activations now
answer to it — no workload-size threshold, no Python-reference fallback — so
a one-element sigmoid runs and stores where a million-element sigmoid does.
Before this, a NumPy tensor of fewer than 32 elements came back in
``PythonStorage``, and everything downstream of an activation in a small
network inherited host residency from it.

Expectations here come from the mathematics and from the range rules the
implementations exist to respect, never from another backend's output. The
two functions are

.. math:: \\sigma(x) = 1/(1 + e^{-x}), \\qquad \\zeta(x) = \\ln(1 + e^{x})

with derivatives \\sigma' = \\sigma(1 - \\sigma) and \\zeta' = \\sigma.
"""

import math
import unittest

import tensors as ts
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.graph.state import reset_graph_state

BACKENDS = ("python", "numpy", "cuda")


def sigmoid_of(x):
    """The logistic function, from whichever side keeps the exponent negative."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus_of(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


class ActivationExecutionTestCase(unittest.TestCase):
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

    def gradient_of(self, function, values, dtype=ts.float64):
        """First-order gradient of ``sum(function(x))``, through the graph."""
        variable = ts.Variable(ts.Tensor(values, dtype=dtype), requires_grad=True)
        (gradient,) = ts.grad(ts.sum(function(variable)), (variable,))
        return gradient


class WhereTheyExecute(ActivationExecutionTestCase):
    """The selection decides, at every size. This is the migration's point."""

    def test_the_forward_stays_on_the_selected_backend_at_every_size(self):
        def body(backend):
            for size in (1, 2, 7, 31, 32, 33, 4096):
                for function in (ts.sigmoid, ts.softplus):
                    produced = function(ts.Tensor([0.5] * size, dtype=ts.float64))
                    self.assertEqual(produced.backend_storage.kind, backend)
                    self.assertEqual(produced.size, size)

        self.for_each_backend(body)

    def test_the_gradient_stays_on_the_selected_backend_at_every_size(self):
        def body(backend):
            for size in (1, 2, 31, 32, 4096):
                for function in (ts.sigmoid, ts.softplus):
                    reset_graph_state()
                    gradient = self.gradient_of(function, [0.5] * size)
                    self.assertEqual(gradient.backend_storage.kind, backend)
                    self.assertEqual(gradient.size, size)

        self.for_each_backend(body)

    def test_a_single_element_matches_the_same_value_in_a_large_tensor(self):
        """Nothing about the result may depend on how much is computed.

        A one-element tensor took the Python reference under NumPy while a
        large one took the NumPy kernel, so this compared two implementations
        as well as two sizes. Now it compares one implementation with itself.
        """

        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                single = function(ts.Tensor([0.7], dtype=ts.float64)).tolist()[0]
                large = function(ts.Tensor([0.7] * 4096, dtype=ts.float64)).tolist()[0]
                self.assertEqual(single, large)

        self.for_each_backend(body)

    def test_a_float32_result_stays_on_the_selected_backend(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                produced = function(ts.Tensor([0.5], dtype=ts.float32))
                self.assertEqual(produced.backend_storage.kind, backend)
                self.assertIs(produced.dtype, ts.float32)

        self.for_each_backend(body)


class TheForwardValues(ActivationExecutionTestCase):
    def test_ordinary_values_are_the_functions_themselves(self):
        def body(backend):
            values = [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0]
            produced = ts.sigmoid(ts.Tensor(values, dtype=ts.float64)).tolist()
            for got, x in zip(produced, values):
                self.assertAlmostEqual(got, sigmoid_of(x), places=12)
            produced = ts.softplus(ts.Tensor(values, dtype=ts.float64)).tolist()
            for got, x in zip(produced, values):
                self.assertAlmostEqual(got, softplus_of(x), places=12)

        self.for_each_backend(body)

    def test_sigmoid_is_a_half_at_zero_and_symmetric_about_it(self):
        def body(backend):
            produced = ts.sigmoid(
                ts.Tensor([0.0, -0.0, 2.5, -2.5], dtype=ts.float64)
            ).tolist()
            self.assertEqual(produced[0], 0.5)
            self.assertEqual(produced[1], 0.5)
            # sigma(-x) = 1 - sigma(x), to the last digit the format holds.
            self.assertAlmostEqual(produced[2] + produced[3], 1.0, places=15)

        self.for_each_backend(body)

    def test_a_signed_zero_gives_the_same_result_as_a_positive_zero(self):
        """Neither function distinguishes them: both are continuous at zero."""

        def body(backend):
            for function, expected in (
                (ts.sigmoid, 0.5),
                (ts.softplus, math.log(2.0)),
            ):
                produced = function(ts.Tensor([0.0, -0.0], dtype=ts.float64)).tolist()
                self.assertEqual(produced[0], produced[1])
                self.assertAlmostEqual(produced[0], expected, places=15)

        self.for_each_backend(body)

    def test_a_large_magnitude_saturates_rather_than_overflowing(self):
        """``exp(800)`` is not representable; both results are.

        This is why each form is evaluated on the side where its exponent is
        negative. Writing sigmoid as ``1/(1 + exp(-x))`` throughout would
        overflow here, and softplus as ``log(1 + exp(x))`` would too, while
        the answers are an ordinary ``1.0`` and ``800.0``.
        """

        def body(backend):
            produced = ts.sigmoid(ts.Tensor([800.0, -800.0], dtype=ts.float64)).tolist()
            self.assertEqual(produced, [1.0, 0.0])
            produced = ts.softplus(
                ts.Tensor([800.0, -800.0], dtype=ts.float64)
            ).tolist()
            self.assertEqual(produced, [800.0, 0.0])

        self.for_each_backend(body)

    def test_infinities_reach_the_limits(self):
        def body(backend):
            produced = ts.sigmoid(
                ts.Tensor([math.inf, -math.inf], dtype=ts.float64)
            ).tolist()
            self.assertEqual(produced, [1.0, 0.0])
            produced = ts.softplus(
                ts.Tensor([math.inf, -math.inf], dtype=ts.float64)
            ).tolist()
            self.assertEqual(produced[0], math.inf)
            self.assertEqual(produced[1], 0.0)

        self.for_each_backend(body)

    def test_nan_propagates(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                produced = function(
                    ts.Tensor([math.nan, 1.0], dtype=ts.float64)
                ).tolist()
                self.assertTrue(math.isnan(produced[0]))
                self.assertFalse(math.isnan(produced[1]))

        self.for_each_backend(body)

    def test_a_subnormal_result_is_not_flushed(self):
        """Both functions reach the binary32 subnormal band around ``x = -90``.

        Section 5.4 requires gradual underflow and does not exempt a format
        crossing, which is why the CUDA kernel converts through PTX in both
        directions rather than with ``astype``.
        """

        def body(backend):
            smallest_normal = 1.1754943508222875e-38
            for function in (ts.sigmoid, ts.softplus):
                produced = function(ts.Tensor([-90.0], dtype=ts.float32)).tolist()[0]
                self.assertGreater(produced, 0.0)
                self.assertLess(produced, smallest_normal)

        self.for_each_backend(body)

    def test_a_subnormal_operand_is_read_rather_than_flushed(self):
        """Both functions are smooth at zero, so a tiny input is near the value there."""

        def body(backend):
            subnormal = 1e-45
            self.assertAlmostEqual(
                ts.sigmoid(ts.Tensor([subnormal], dtype=ts.float32)).tolist()[0],
                0.5,
                places=6,
            )
            self.assertAlmostEqual(
                ts.softplus(ts.Tensor([subnormal], dtype=ts.float32)).tolist()[0],
                math.log(2.0),
                places=6,
            )

        self.for_each_backend(body)


class TheGradientValues(ActivationExecutionTestCase):
    def test_the_sigmoid_derivative_is_the_product_rule(self):
        def body(backend):
            values = [-2.0, -0.5, 0.0, 0.5, 2.0]
            produced = self.gradient_of(ts.sigmoid, values).tolist()
            for got, x in zip(produced, values):
                s = sigmoid_of(x)
                self.assertAlmostEqual(got, s * (1.0 - s), places=12)

        self.for_each_backend(body)

    def test_the_softplus_derivative_is_the_sigmoid(self):
        def body(backend):
            values = [-2.0, -0.5, 0.0, 0.5, 2.0]
            produced = self.gradient_of(ts.softplus, values).tolist()
            for got, x in zip(produced, values):
                self.assertAlmostEqual(got, sigmoid_of(x), places=12)

        self.for_each_backend(body)

    def test_the_sigmoid_derivative_survives_a_saturated_sigmoid(self):
        """At ``x = 20`` the sigmoid rounds to one but its slope is not zero.

        ``s * (1 - s)`` gives exactly ``0.0`` there, because ``1 - s`` is a
        subtraction from a rounded one. The kernels evaluate
        ``z / (1 + z)**2`` instead, which keeps the digits the format has.
        """

        def body(backend):
            # z / (1 + z)**2 at z = exp(-20), to binary64: distinctly below
            # exp(-20) itself, and nothing like the 0.0 that s * (1 - s) gives.
            expected = 2.061153613941849e-09
            produced = self.gradient_of(ts.sigmoid, [20.0, -20.0]).tolist()
            for got in produced:
                self.assertGreater(got, 0.0)
                self.assertAlmostEqual(got / expected, 1.0, places=12)
            # It is symmetric in |x|, so the two tails agree exactly.
            self.assertEqual(produced[0], produced[1])

        self.for_each_backend(body)

    def test_a_saturated_gradient_is_zero_rather_than_undefined(self):
        def body(backend):
            produced = self.gradient_of(ts.sigmoid, [800.0, -800.0]).tolist()
            self.assertEqual(produced, [0.0, 0.0])
            produced = self.gradient_of(ts.softplus, [800.0, -800.0]).tolist()
            self.assertEqual(produced, [1.0, 0.0])

        self.for_each_backend(body)

    def test_a_nan_primal_gives_a_nan_gradient(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                reset_graph_state()
                produced = self.gradient_of(function, [math.nan, 1.0]).tolist()
                self.assertTrue(math.isnan(produced[0]))
                self.assertFalse(math.isnan(produced[1]))

        self.for_each_backend(body)

    def test_an_infinite_primal_gives_a_finite_gradient(self):
        def body(backend):
            produced = self.gradient_of(ts.sigmoid, [math.inf, -math.inf]).tolist()
            self.assertEqual(produced, [0.0, 0.0])
            produced = self.gradient_of(ts.softplus, [math.inf, -math.inf]).tolist()
            self.assertEqual(produced, [1.0, 0.0])

        self.for_each_backend(body)

    def test_the_gradient_scales_with_the_upstream(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                reset_graph_state()
                variable = ts.Variable(
                    ts.Tensor([0.5, -0.5], dtype=ts.float64), requires_grad=True
                )
                doubled = ts.grad(
                    function(variable),
                    variable,
                    grad_outputs=ts.Tensor([2.0, 2.0], dtype=ts.float64),
                )
                reset_graph_state()
                variable = ts.Variable(
                    ts.Tensor([0.5, -0.5], dtype=ts.float64), requires_grad=True
                )
                once = ts.grad(
                    function(variable),
                    variable,
                    grad_outputs=ts.Tensor([1.0, 1.0], dtype=ts.float64),
                )
                for two, one in zip(doubled.tolist(), once.tolist()):
                    self.assertAlmostEqual(two, 2.0 * one, places=15)

        self.for_each_backend(body)


class DtypesAndShapes(ActivationExecutionTestCase):
    def test_a_floating_dtype_is_preserved(self):
        def body(backend):
            for dtype in (ts.float32, ts.float64):
                for function in (ts.sigmoid, ts.softplus):
                    self.assertIs(function(ts.Tensor([0.5], dtype=dtype)).dtype, dtype)

        self.for_each_backend(body)

    def test_an_integer_operand_promotes_to_float64(self):
        """Neither function has integral values to return."""

        def body(backend):
            for dtype in (ts.int32, ts.int64):
                produced = ts.sigmoid(ts.Tensor([-2, 0, 3], dtype=dtype))
                self.assertIs(produced.dtype, ts.float64)
                self.assertEqual(produced.backend_storage.kind, backend)
                for got, x in zip(produced.tolist(), (-2.0, 0.0, 3.0)):
                    self.assertAlmostEqual(got, sigmoid_of(x), places=12)
                produced = ts.softplus(ts.Tensor([-2, 0, 3], dtype=dtype))
                self.assertIs(produced.dtype, ts.float64)
                self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_the_shape_is_the_operands(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                produced = function(
                    ts.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=ts.float64)
                )
                self.assertEqual(tuple(produced.shape), (2, 3))
                self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_a_non_contiguous_operand_is_read_by_its_logical_values(self):
        def body(backend):
            source = ts.Tensor([float(i) for i in range(8)], dtype=ts.float64)
            strided = source[::2]
            self.assertEqual(strided.tolist(), [0.0, 2.0, 4.0, 6.0])
            for function, reference in (
                (ts.sigmoid, sigmoid_of),
                (ts.softplus, softplus_of),
            ):
                produced = function(strided)
                self.assertEqual(produced.backend_storage.kind, backend)
                for got, x in zip(produced.tolist(), (0.0, 2.0, 4.0, 6.0)):
                    self.assertAlmostEqual(got, reference(x), places=12)

        self.for_each_backend(body)

    def test_an_empty_tensor_gives_an_empty_result(self):
        def body(backend):
            for function in (ts.sigmoid, ts.softplus):
                produced = function(ts.Tensor([], dtype=ts.float64, shape=(0,)))
                self.assertEqual(produced.tolist(), [])
                self.assertEqual(produced.backend_storage.kind, backend)

        self.for_each_backend(body)

    def test_a_mismatched_gradient_is_refused_by_the_operation(self):
        """Shape and dtype agreement are Tensor semantics, settled in backward."""
        from tensors.operations.activations.sigmoid import Sigmoid
        from tensors.operations.activations.softplus import Softplus

        def body(backend):
            for operation in (Sigmoid(), Softplus()):
                with self.assertRaisesRegex(ValueError, "does not match value shape"):
                    operation.backward(
                        ts.Tensor([1.0, 2.0], dtype=ts.float64),
                        ts.Tensor([1.0, 2.0, 3.0], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )
                with self.assertRaisesRegex(ValueError, "does not match value dtype"):
                    operation.backward(
                        ts.Tensor([1.0], dtype=ts.float32),
                        ts.Tensor([1.0], dtype=ts.float64),
                        needs_input_grad=(True,),
                    )

        self.for_each_backend(body)


class TheBackendBoundary(ActivationExecutionTestCase):
    """What the dispatchers own, and what they refuse."""

    def test_a_backend_that_declines_reports_it_rather_than_falling_back(self):
        from unittest.mock import patch

        from tensors.backend import dispatch as backend_dispatch

        def body(backend):
            if backend == "python":
                self.skipTest("the reference kernel is the one that would be used")
            import tensors.backend.numpy.kernels as numpy_kernels
            import tensors.backend.cuda.kernels as cuda_kernels

            kernels = numpy_kernels if backend == "numpy" else cuda_kernels
            for name, function in (
                ("sigmoid", backend_dispatch.execute_sigmoid),
                ("softplus", backend_dispatch.execute_softplus),
            ):
                with patch.object(kernels, name, return_value=None):
                    from tensors.backend.loading import _clear_backend_kernel_cache

                    _clear_backend_kernel_cache()
                    with self.assertRaises(BackendOperationUnsupportedError) as raised:
                        function(
                            ts.Tensor([0.5], dtype=ts.float64),
                            dtype=ts.float64,
                            output_shape=(1,),
                        )
                    message = str(raised.exception)
                    self.assertIn(backend, message)
                    self.assertIn(name, message)
                    self.assertIn("float64", message)
            from tensors.backend.loading import _clear_backend_kernel_cache

            _clear_backend_kernel_cache()

        self.for_each_backend(body)

    def test_an_operand_from_another_backend_is_refused(self):
        from tensors.backend.config import BackendMismatchError
        from tensors.backend import dispatch as backend_dispatch

        if "numpy" not in ts.available_backends():
            self.skipTest("NumPy is not available here")
        with ts.use_backend("python"):
            host = ts.Tensor([0.5], dtype=ts.float64)
        with ts.use_backend("numpy"):
            with self.assertRaises(BackendMismatchError):
                backend_dispatch.execute_sigmoid(
                    host, dtype=ts.float64, output_shape=(1,)
                )
            with self.assertRaises(BackendMismatchError):
                backend_dispatch.execute_softplus(
                    host, dtype=ts.float64, output_shape=(1,)
                )

    def test_a_fused_chain_agrees_with_the_ordinary_path(self):
        """A fused run changes the execution plan, never the result.

        Both activations are fusible, so a CUDA chain containing one can
        execute as a single launch instead of as separate operations. The
        fused expression and the kernel are two statements of the same
        function and have to agree element for element.
        """
        if "cuda" not in ts.available_backends():
            self.skipTest("fusion is a CUDA execution plan")
        from unittest.mock import patch

        import tensors.backend as backend_state
        import tensors.backend.cuda.kernels as cuda_kernels

        values = [-90.0, -20.0, -1.0, 0.0, 0.5, 20.0, 90.0, 800.0]
        for dtype in (ts.float64, ts.float32):
            for name, function in (("sigmoid", ts.sigmoid), ("softplus", ts.softplus)):
                with self.subTest(dtype=dtype.name, activation=name):
                    reset_graph_state()
                    with ts.use_backend("cuda"):
                        ordinary = function(ts.Tensor(values, dtype=dtype)).tolist()

                        size = 4096
                        repeated = [values[i % len(values)] for i in range(size)]
                        variable = ts.Variable(
                            ts.Tensor(repeated, dtype=dtype), requires_grad=False
                        )
                        output = function(variable * 1.0)
                        computation = ts.graph.Computation(output)
                        with patch.object(
                            cuda_kernels,
                            "fused_elementwise",
                            wraps=cuda_kernels.fused_elementwise,
                        ) as fused:
                            backend_state._clear_backend_kernel_cache()
                            result = computation.forward()
                        self.assertGreater(fused.call_count, 0)
                        produced = result.tolist()[: len(values)]
                    for got, expected in zip(produced, ordinary):
                        if math.isnan(expected):
                            self.assertTrue(math.isnan(got))
                        else:
                            self.assertEqual(got, expected)
                    reset_graph_state()

    def test_the_dispatchers_consult_no_workload_policy(self):
        """The threshold is what made a small tensor change backends."""
        import inspect

        from tensors.backend.dispatch.elementwise import (
            sigmoid,
            sigmoid_gradient,
            softplus,
            softplus_gradient,
        )

        for module in (sigmoid, sigmoid_gradient, softplus, softplus_gradient):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("_array_work_is_large_enough", source)
                self.assertNotIn("from tensors.backend.policy", source)
                # No reference kernel is imported to fall back to: the
                # selected backend's package is the only one loaded.
                self.assertNotIn("kernels.elementwise", source)
                self.assertIn("validate_backend_residency", source)
                self.assertIn("BackendOperationUnsupportedError", source)


if __name__ == "__main__":
    unittest.main()
