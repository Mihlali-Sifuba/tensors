"""Where the ``+``, ``-`` and ``*`` vector-Jacobian products execute.

`docs/backends.md`, *Execution requirements*, makes the backend selection an
execution requirement. These tests ask where a reverse pass ran, not whether
its numbers are right: correctness of the gradients themselves is covered in
`tests/operations/arithmetic` and `tests/backend/test_reductions.py`.

Two details make the question harder than it looks, and both shaped these
tests.

First, the upstream seed. ``ts.backward`` with no gradient builds one with
``ones``, whose dispatch is a creation operation outside this contract and
still keeps its workload policy. A small seed therefore arrives in
``PythonStorage`` whatever backend is selected. Where a VJP has nothing to
compute — adding two same-shaped operands hands the gradient straight back —
the result carries the seed's storage, and that would look like a backend
switch while no computation happened at all. :func:`resident_seed` builds a
seed on the selected backend so the question being asked is the VJP's.

Second, ``sum_to_shape`` only runs when a gradient must actually be reduced,
so the same-shape and broadcast cases exercise different code. Both appear
below.
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

#: Well below the 32-element threshold that used to divert these to Python.
SMALL = (1, 4, 31)
LARGE = (64, 4096)

OPERATIONS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}


def storage_name(tensor) -> str:
    return type(tensor._storage).__name__


def resident_seed(shape, value=1.0):
    """Build an upstream gradient that already lives on the selected backend.

    Arithmetic is bound to the selection, so one arithmetic operation places
    the seed without relying on the creation policy this contract does not
    cover.
    """
    return ts.full(shape, 0.0, dtype=ts.float64) + value


def backward_storages(backend, operation, size, broadcast=False):
    """Run one reverse pass and report where both gradients ended up."""
    with ts.use_backend(backend):
        left = ts.Variable(ts.full((2, size), 1.5, dtype=ts.float64))
        right = ts.Variable(
            ts.full((1, size) if broadcast else (2, size), 2.5, dtype=ts.float64)
        )
        output = operation(left, right)
        ts.backward(output, resident_seed(output.data.shape))
        return storage_name(left.grad), storage_name(right.grad)


class VjpExecutesOnTheSelectedBackend(unittest.TestCase):
    """Every size, every operation, both shapes."""

    def _assert_runs_on(self, selection, expected=None):
        """``selection`` is what the caller selects; ``expected`` where it runs."""
        expected = STORAGE_FOR[expected or selection]
        for symbol, operation in OPERATIONS.items():
            for size in SMALL + LARGE:
                for broadcast in (False, True):
                    with self.subTest(op=symbol, size=size, broadcast=broadcast):
                        storages = backward_storages(
                            selection, operation, size, broadcast
                        )
                        self.assertEqual(storages, (expected, expected))

    def test_python_selection_executes_on_python(self):
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
        with patch.object(config, "_numpy_available", return_value=False):
            with ts.use_backend("auto"):
                self.assertEqual(ts.get_backend(), "python")
            for symbol, operation in OPERATIONS.items():
                for size in SMALL:
                    with self.subTest(op=symbol, size=size):
                        self.assertEqual(
                            backward_storages("auto", operation, size),
                            ("PythonStorage", "PythonStorage"),
                        )


@requires_numpy
class SmallVjpsReachTheProviderKernel(unittest.TestCase):
    """The kernel is called, rather than the size sending work elsewhere."""

    KERNEL_FOR = {
        "+": "sum_to_shape",
        "-": "sum_to_shape",
        "*": "sum_products_to_shape",
    }

    def test_a_four_element_reverse_pass_calls_the_numpy_kernel(self):
        for symbol, operation in OPERATIONS.items():
            kernel_name = self.KERNEL_FOR[symbol]
            with self.subTest(op=symbol):
                original = getattr(numpy_kernels, kernel_name)
                with patch.object(numpy_kernels, kernel_name, wraps=original) as kernel:
                    backend_state._clear_backend_kernel_cache()
                    backward_storages("numpy", operation, 4, broadcast=symbol != "*")
                backend_state._clear_backend_kernel_cache()
                self.assertGreaterEqual(kernel.call_count, 1)

    def test_a_four_element_subtraction_negates_on_numpy(self):
        original = numpy_kernels.negate
        with patch.object(numpy_kernels, "negate", wraps=original) as negate:
            backend_state._clear_backend_kernel_cache()
            backward_storages("numpy", OPERATIONS["-"], 4)
        backend_state._clear_backend_kernel_cache()
        self.assertGreaterEqual(negate.call_count, 1)


class ADecliningBackendRaises(unittest.TestCase):
    """A kernel that cannot answer is reported, never quietly replaced."""

    def _assert_raises_when_declining(self, backend, kernel_name, operation, **kwargs):
        import importlib

        kernels = importlib.import_module(f"tensors.backend.{backend}.kernels")
        with patch.object(kernels, kernel_name, return_value=None):
            backend_state._clear_backend_kernel_cache()
            try:
                with self.assertRaises(ts.BackendOperationUnsupportedError) as raised:
                    backward_storages(backend, operation, 4, **kwargs)
            finally:
                backend_state._clear_backend_kernel_cache()
        message = str(raised.exception)
        self.assertIn(backend, message)
        self.assertIn(kernel_name, message)

    @requires_numpy
    def test_numpy_declining_sum_to_shape_raises(self):
        for symbol in ("+", "-"):
            with self.subTest(op=symbol):
                self._assert_raises_when_declining(
                    "numpy", "sum_to_shape", OPERATIONS[symbol], broadcast=True
                )

    @requires_numpy
    def test_numpy_declining_sum_products_to_shape_raises(self):
        self._assert_raises_when_declining(
            "numpy", "sum_products_to_shape", OPERATIONS["*"]
        )

    @requires_numpy
    def test_numpy_declining_negate_raises(self):
        self._assert_raises_when_declining("numpy", "negate", OPERATIONS["-"])

    @requires_cuda
    def test_cuda_declining_sum_products_to_shape_raises(self):
        self._assert_raises_when_declining(
            "cuda", "sum_products_to_shape", OPERATIONS["*"]
        )


@requires_numpy
class GradientsAreUnchanged(unittest.TestCase):
    """Relocating the work must not have moved any number."""

    def test_small_gradients_match_the_python_backend(self):
        def gradients(backend, operation, broadcast):
            with ts.use_backend(backend):
                left = ts.Variable(ts.Tensor([[1.5, -2.25, 0.5, 3.0]] * 2))
                right = ts.Variable(
                    ts.Tensor([[0.25, 4.0, -1.5, 2.0]] * (1 if broadcast else 2))
                )
                output = operation(left, right)
                ts.backward(output, resident_seed(output.data.shape))
                return left.grad.tolist(), right.grad.tolist()

        for symbol, operation in OPERATIONS.items():
            for broadcast in (False, True):
                with self.subTest(op=symbol, broadcast=broadcast):
                    self.assertEqual(
                        gradients("numpy", operation, broadcast),
                        gradients("python", operation, broadcast),
                    )


if __name__ == "__main__":
    unittest.main()
