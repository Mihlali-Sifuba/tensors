"""Arithmetic execution location, residency and fusion equivalence.

`docs/backends.md`, *Execution requirements*: explicit selection is an
execution requirement, not a preference. `docs/autodiff.md` requires an
optimisation to preserve the numerical result of the unoptimised sequence.

Section 9.4 of the arithmetic specification separates these from the semantic
conformance tests: here the question is *where* arithmetic ran and *what it
produced relative to an unfused run*, not whether a value is correct.
"""

import unittest
from unittest.mock import patch

import tensors as ts
from tensors.backend import loading
from tensors.graph import Computation

from . import _spec
from ._support import BACKENDS, ArithmeticTestCase, tensor

ACCELERATED = tuple(name for name in BACKENDS if name != "python")
INTEGERS = tuple(_spec.INTEGER_DTYPES)

#: Large enough that no workload policy would route it to Python for size.
SIZE = 100_000

OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

STORAGE_FOR = {
    "numpy": "NumPyStorage",
    "cuda": "CudaStorage",
    "python": "PythonStorage",
}


def storage_name(result) -> str:
    return type(result.backend_storage).__name__


class ExplicitSelectionTests(ArithmeticTestCase):
    """Supported arithmetic executes on the backend that was selected."""

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_integer_arithmetic_stays_on_the_selected_backend(self):
        for backend in ACCELERATED:
            for dtype_name in INTEGERS:
                for symbol, operation in OPERATIONS.items():
                    with self.subTest(backend=backend, dtype=dtype_name, op=symbol):
                        with ts.use_backend(backend):
                            left = ts.full((SIZE,), 3, dtype=getattr(ts, dtype_name))
                            right = ts.full((SIZE,), 2, dtype=getattr(ts, dtype_name))
                            result = operation(left, right)
                            self.assertEqual(storage_name(result), STORAGE_FOR[backend])

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_floating_arithmetic_stays_on_the_selected_backend(self):
        for backend in ACCELERATED:
            for dtype_name in _spec.FLOAT_DTYPES:
                for symbol, operation in OPERATIONS.items():
                    with self.subTest(backend=backend, dtype=dtype_name, op=symbol):
                        with ts.use_backend(backend):
                            left = ts.full((SIZE,), 1.5, dtype=getattr(ts, dtype_name))
                            right = ts.full((SIZE,), 2.0, dtype=getattr(ts, dtype_name))
                            result = operation(left, right)
                            self.assertEqual(storage_name(result), STORAGE_FOR[backend])

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_floating_overflow_does_not_leave_the_backend(self):
        """Overflow is a result, not a reason to fall back (section 5.1)."""
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    big = ts.full((SIZE,), 3.0e38, dtype=ts.float32)
                    result = big + big
                    self.assertEqual(storage_name(result), STORAGE_FOR[backend])
                    self.assertEqual(result.tolist()[0], float("inf"))

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_integer_overflow_does_not_leave_the_backend(self):
        for backend in ACCELERATED:
            for dtype_name in INTEGERS:
                low, high = _spec.integer_range(dtype_name)
                with self.subTest(backend=backend, dtype=dtype_name):
                    with ts.use_backend(backend):
                        operand = ts.full((SIZE,), high, dtype=getattr(ts, dtype_name))
                        one = ts.full((SIZE,), 1, dtype=getattr(ts, dtype_name))
                        result = operand + one
                        self.assertEqual(storage_name(result), STORAGE_FOR[backend])
                        self.assertEqual(
                            int(result.tolist()[0]), _spec.wrap(high + 1, dtype_name)
                        )

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_division_by_zero_does_not_leave_the_backend(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    numerator = ts.full((SIZE,), 1.0, dtype=ts.float64)
                    denominator = ts.full((SIZE,), 0.0, dtype=ts.float64)
                    result = numerator / denominator
                    self.assertEqual(storage_name(result), STORAGE_FOR[backend])
                    self.assertEqual(result.tolist()[0], float("inf"))

    @unittest.skipUnless(ACCELERATED, "no accelerated backend installed")
    def test_a_small_tensor_still_uses_the_selected_backend(self):
        """Workload policy must not override explicit selection."""
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    left = ts.full((4,), 1.5, dtype=ts.float32)
                    result = left + left
                    self.assertEqual(storage_name(result), STORAGE_FOR[backend])


class ResidencyTests(ArithmeticTestCase):
    """Results stay where the backend puts them."""

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_cuda_arithmetic_results_are_device_resident(self):
        for dtype_name in INTEGERS + _spec.FLOAT_DTYPES:
            with self.subTest(dtype=dtype_name):
                with ts.use_backend("cuda"):
                    left = ts.full((SIZE,), 2, dtype=getattr(ts, dtype_name))
                    result = left + left
                    self.assertEqual(storage_name(result), "CudaStorage")


class FusionEquivalenceTests(ArithmeticTestCase):
    """A fused plan must produce the unfused result (autodiff.md).

    Size matters here. CUDA fusion has a work threshold, so a small tensor
    compiles no fused kernel and a test built on one proves nothing about
    fusion. :data:`FUSED_SIZE` is above that threshold, and
    :meth:`test_the_cuda_fusion_kernel_is_actually_reached` fails if the
    threshold moves past it and quietly turns the rest into unfused runs.
    """

    #: Above tensors.backend.policy._CUDA_FUSION_MIN_WORK.
    FUSED_SIZE = 16_384

    EXPRESSIONS = {
        "multiply_add": lambda a, b, c: a * b + c,
        "chain": lambda a, b, c: (a + b) * c,
        "divide_add": lambda a, b, c: a / b + c,
        "subtract_divide": lambda a, b, c: (a - b) / c,
    }

    def _operands(self, dtype_name, size, third=None):
        values = [0.5, 38.24823, -2.25, 1.0000001, 3.14159265358979, -7.7777777]
        repeated = (values * (size // len(values) + 1))[:size]
        shifted = list(reversed(repeated))
        other = [v + 0.125 for v in repeated] if third is None else third * size
        return (
            tensor(dtype_name, repeated),
            tensor(dtype_name, shifted),
            tensor(dtype_name, other[:size]),
        )

    def _fused(self, expression, operands, requires_grad=False):
        """Compile the expression and replay it, which is where fusion runs.

        Building a Variable expression evaluates it eagerly; only a compiled
        Computation reaches the fused kernel.
        """
        variables = [ts.Variable(x, requires_grad=requires_grad) for x in operands]
        program = Computation(expression(*variables))
        return program.forward(), program, variables

    @unittest.skipUnless("cuda" in BACKENDS, "CUDA backend is not installed")
    def test_the_cuda_fusion_kernel_is_actually_reached(self):
        """Guards every other test in this class."""
        import tensors.backend.cuda.kernels as cuda_backend

        with ts.use_backend("cuda"):
            operands = self._operands("float32", self.FUSED_SIZE)
            with patch.object(
                cuda_backend,
                "fused_elementwise",
                wraps=cuda_backend.fused_elementwise,
            ) as fused:
                loading._clear_backend_kernel_cache()
                self._fused(self.EXPRESSIONS["multiply_add"], operands)[0]
            self.assertTrue(
                fused.called,
                "CUDA fusion did not run; these tests no longer test fusion",
            )

    def test_fused_and_unfused_agree_bitwise(self):
        """Executed through Variables, where fusion applies."""
        for dtype_name in _spec.FLOAT_DTYPES:
            for name, expression in self.EXPRESSIONS.items():
                for backend in BACKENDS:
                    with self.subTest(dtype=dtype_name, expr=name, backend=backend):
                        with ts.use_backend(backend):
                            operands = self._operands(dtype_name, self.FUSED_SIZE)
                            stepwise = expression(*operands)
                            fused = self._fused(expression, operands)[0]
                            self.assertFloatBitsEqual(
                                fused, stepwise.tolist(), dtype_name
                            )

    def test_a_fused_division_by_zero_gives_the_unfused_result(self):
        """A fused plan may not raise where the sequence returns an infinity.

        Section 7.2 makes a zero denominator a result. The fused CUDA kernel
        used to test every denominator and raise, so the same expression gave
        an infinity eagerly and a ZeroDivisionError once compiled.
        """
        expression = self.EXPRESSIONS["divide_add"]
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        operands = self._operands(
                            dtype_name, self.FUSED_SIZE, third=[0.0]
                        )
                        a, b, c = operands
                        # Divide by the all-zero tensor, then add.
                        stepwise = (a / c) + b
                        fused = self._fused(lambda x, y, z: x / z + y, operands)[0]
                        self.assertFloatBitsEqual(fused, stepwise.tolist(), dtype_name)

    def test_a_fused_multiply_add_preserves_both_roundings(self):
        """No FMA contraction: round(a*b) then round(+c).

        Operands are chosen so a single-rounded fused multiply-add differs
        from two roundings in a good fraction of elements, at a size that
        compiles a fused kernel.
        """
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        operands = self._operands(dtype_name, self.FUSED_SIZE)
                        a, b, c = operands
                        expected = ((a * b) + c).tolist()
                        fused = self._fused(self.EXPRESSIONS["multiply_add"], operands)[
                            0
                        ]
                        self.assertFloatBitsEqual(fused, expected, dtype_name)

    def test_a_fused_backward_pass_matches_the_unfused_gradients(self):
        """The backward kernel is compiled with the same guarantees."""
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        operands = self._operands(dtype_name, self.FUSED_SIZE)
                        stepwise = []
                        for index in range(3):
                            variables = [
                                ts.Variable(x, requires_grad=(i == index))
                                for i, x in enumerate(operands)
                            ]
                            product = variables[0] * variables[1]
                            ts.backward(ts.sum(product + variables[2]))
                            stepwise.append(variables[index].grad.tolist())

                        _, program, variables = self._fused(
                            self.EXPRESSIONS["multiply_add"],
                            operands,
                            requires_grad=True,
                        )
                        program.backward(ts.full(operands[0].shape, 1.0))
                        for index in range(3):
                            self.assertFloatBitsEqual(
                                variables[index].grad, stepwise[index], dtype_name
                            )


class GraphReplayTests(ArithmeticTestCase):
    """Replay produces what eager execution produced."""

    OPERANDS = ([1.5, 2.5, -0.25, 1e-30], [0.1, -2.0, 4.0, 3.0])

    def test_replay_matches_eager_arithmetic(self):
        """Tensor arithmetic, the first Variable pass and a replay agree."""
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        a = tensor(dtype_name, self.OPERANDS[0])
                        b = tensor(dtype_name, self.OPERANDS[1])
                        stepwise = ((a + b) * a).tolist()

                        x = ts.Variable(a, requires_grad=False)
                        y = ts.Variable(b, requires_grad=False)
                        output = (x + y) * x
                        self.assertFloatBitsEqual(output.data, stepwise, dtype_name)

                        program = Computation(output)
                        self.assertFloatBitsEqual(
                            program.forward(), stepwise, dtype_name
                        )

    def test_replay_after_new_inputs_matches_eager(self):
        """A second replay over changed inputs still matches eager arithmetic."""
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        x = ts.Variable(
                            tensor(dtype_name, self.OPERANDS[0]), requires_grad=False
                        )
                        y = ts.Variable(
                            tensor(dtype_name, self.OPERANDS[1]), requires_grad=False
                        )
                        program = Computation((x + y) * x)
                        program.forward()

                        replacement = tensor(dtype_name, [2.0, -3.5, 1e-38, 0.125])
                        x.data = replacement
                        expected = ((replacement + y.data) * replacement).tolist()
                        self.assertFloatBitsEqual(
                            program.forward(), expected, dtype_name
                        )


class BackendConsistencyTests(ArithmeticTestCase):
    """The same expression on every backend gives the same bits."""

    def test_arithmetic_agrees_across_backends(self):
        if len(BACKENDS) < 2:
            self.skipTest("needs more than one backend")
        values = [0.1, -2.5, 1e-40, 3.0e38, 0.0, -0.0]
        for dtype_name in _spec.FLOAT_DTYPES:
            for symbol, operation in OPERATIONS.items():
                results = {}
                for backend in BACKENDS:
                    with ts.use_backend(backend):
                        left = tensor(dtype_name, values)
                        right = tensor(dtype_name, list(reversed(values)))
                        results[backend] = operation(left, right).tolist()
                reference_backend = BACKENDS[0]
                for backend in BACKENDS[1:]:
                    with self.subTest(dtype=dtype_name, op=symbol, backend=backend):
                        self.assertFloatBitsEqual(
                            _AsResult(results[backend], dtype_name),
                            results[reference_backend],
                            dtype_name,
                        )


class _AsResult:
    """Adapt a plain list to the shape assertFloatBitsEqual expects."""

    def __init__(self, values, dtype_name):
        self._values = values
        self.dtype = getattr(ts, dtype_name)

    def tolist(self):
        return self._values


if __name__ == "__main__":
    unittest.main()
