"""Arithmetic execution location, residency and fusion equivalence.

`docs/backends.md`, *Execution requirements*: explicit selection is an
execution requirement, not a preference. `docs/autodiff.md` requires an
optimisation to preserve the numerical result of the unoptimised sequence.

Section 9.4 of the arithmetic specification separates these from the semantic
conformance tests: here the question is *where* arithmetic ran and *what it
produced relative to an unfused run*, not whether a value is correct.
"""

import unittest

import tensors as ts

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
    return type(result._storage).__name__


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
    """A fused plan must produce the unfused result (autodiff.md)."""

    EXPRESSIONS = {
        "multiply_add": lambda a, b, c: a * b + c,
        "chain": lambda a, b, c: (a + b) * c,
        "divide_add": lambda a, b, c: a / b + c,
    }

    def _operands(self, dtype_name, size):
        values = [0.5, 38.24823, -2.25, 1.0000001]
        repeated = (values * (size // len(values) + 1))[:size]
        shifted = list(reversed(repeated))
        other = [v + 0.125 for v in repeated]
        return (
            tensor(dtype_name, repeated),
            tensor(dtype_name, shifted),
            tensor(dtype_name, other),
        )

    def test_fused_and_unfused_agree_bitwise(self):
        """Executed through Variables, where fusion applies."""
        for dtype_name in _spec.FLOAT_DTYPES:
            for name, expression in self.EXPRESSIONS.items():
                for backend in BACKENDS:
                    with self.subTest(dtype=dtype_name, expr=name, backend=backend):
                        with ts.use_backend(backend):
                            a, b, c = self._operands(dtype_name, 512)
                            stepwise = expression(a, b, c)
                            variables = [
                                ts.Variable(x, requires_grad=False) for x in (a, b, c)
                            ]
                            fused = expression(*variables)
                            self.assertFloatBitsEqual(
                                fused.data, stepwise.tolist(), dtype_name
                            )

    def test_a_fused_multiply_add_preserves_both_roundings(self):
        """No FMA contraction: round(a*b) then round(+c)."""
        a = tensor("float32", [38.24823])
        b = tensor("float32", [280.5834])
        c = tensor("float32", [-0.4536956])
        product = a * b
        expected = (product + c).tolist()
        variables = [ts.Variable(x, requires_grad=False) for x in (a, b, c)]
        fused = variables[0] * variables[1] + variables[2]
        self.assertFloatBitsEqual(fused.data, expected, "float32")


class GraphReplayTests(ArithmeticTestCase):
    """Replay produces what eager execution produced."""

    def test_replay_matches_eager_arithmetic(self):
        for dtype_name in _spec.FLOAT_DTYPES:
            for backend in BACKENDS:
                with self.subTest(dtype=dtype_name, backend=backend):
                    with ts.use_backend(backend):
                        a = tensor(dtype_name, [1.5, 2.5, -0.25, 1e-30])
                        b = tensor(dtype_name, [0.1, -2.0, 4.0, 3.0])
                        eager = (a + b) * a
                        node = ts.graph.node.VariableNode()
                        output = (node + ts.Variable(b, requires_grad=False)) * node
                        program = ts.graph.Computation(output, boundaries=(node,))
                        replayed = program(a)
                        self.assertFloatBitsEqual(
                            (
                                replayed
                                if not hasattr(replayed, "data")
                                else replayed.data
                            ),
                            eager.tolist(),
                            dtype_name,
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
