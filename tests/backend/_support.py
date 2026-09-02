"""Shared scaffolding for the backend test modules.

``tests/backend`` splits the former ``tests/test_backends.py`` by feature
rather than by backend, so a kernel and the tests covering it sit in parallel
trees. The availability guards and parity helpers used by several of those
modules live here.
"""

import unittest

import tensors as ts


requires_numpy = unittest.skipUnless(
    "numpy" in ts.available_backends(),
    "NumPy is not installed",
)

requires_cuda = unittest.skipUnless(
    "cuda" in ts.available_backends(),
    "CUDA is not available",
)


class BackendTestCase(unittest.TestCase):
    """Restore the process backend selection after every test."""

    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)


class NumPyParityTestCase(BackendTestCase):
    """Compare NumPy kernel results against the Python reference."""

    def _matmul(self, backend, left, right):
        with ts.use_backend(backend):
            return ts.matmul(left, right)

    def assertBackendParity(self, left, right):
        expected = self._matmul("python", left, right)
        actual = self._matmul("numpy", left, right)

        self.assertEqual(actual.shape, expected.shape)
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.tolist(), expected.tolist())

    def _evaluate(self, backend, function):
        with ts.use_backend(backend):
            result = function()
            return result.data if isinstance(result, ts.Variable) else result

    def assertOperationParity(self, function):
        expected = self._evaluate("python", function)
        actual = self._evaluate("numpy", function)

        self.assertEqual(actual.shape, expected.shape)
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.tolist(), expected.tolist())
