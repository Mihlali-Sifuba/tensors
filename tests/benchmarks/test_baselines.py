"""A baseline has to hold the same values as the thing it is compared against.

The ``provider`` rung of every ladder claims to be the same computation as
the ``public`` rung with the library called directly. That claim is only
worth anything if both sides start from identical inputs, so these check it
rather than assuming it.
"""

import unittest

import tensors as ts

from benchmarks import baselines
from benchmarks.case import Unsupported
from benchmarks.inputs import tensor

ACCELERATED = tuple(
    backend for backend in ts.available_backends() if backend != "python"
)


class BaselineSelectionTests(unittest.TestCase):
    """Which library stands in for which backend."""

    def test_each_accelerated_backend_has_a_baseline(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                self.assertEqual(
                    baselines.module(backend).__name__,
                    baselines.BASELINE_MODULES[backend],
                )

    def test_the_python_backend_has_none(self):
        """Its reference kernels are the implementation, not a comparison."""
        with self.assertRaises(Unsupported):
            baselines.module("python")

    def test_an_unknown_backend_is_declined_rather_than_guessed(self):
        with self.assertRaises(Unsupported):
            baselines.module("metal")


class BaselineValueTests(unittest.TestCase):
    """A baseline array and a Tensor hold the same numbers."""

    SHAPES = ((1,), (5,), (3, 4))

    def _compare(self, backend, shape, **options):
        with ts.use_backend(backend):
            expected = tensor(shape, **options).tolist()
        native = baselines.array(backend, shape, **options)
        self.assertEqual(native.reshape(-1).tolist(), expected)

    def test_a_ramp_matches(self):
        for backend in ACCELERATED:
            for shape in self.SHAPES:
                with self.subTest(backend=backend, shape=shape):
                    self._compare(backend, shape, kind="ramp")

    def test_a_mixed_sign_pattern_matches(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                self._compare(backend, (6,), kind="mixed")

    def test_a_constant_matches(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                self._compare(backend, (4,), kind="constant", value=2.5)

    def test_an_integer_dtype_matches(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                self._compare(backend, (5,), dtype_name="int64", kind="ramp")

    def test_an_unknown_kind_is_refused(self):
        for backend in ACCELERATED:
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "unknown value kind"):
                    baselines.array(backend, (2,), kind="random")

    def test_the_two_baselines_agree_with_each_other(self):
        """Both are the same comparison, so they cannot hold different values."""
        if len(ACCELERATED) < 2:
            self.skipTest("needs both NumPy and CUDA installed")
        values = [
            baselines.array(backend, (7,), kind="mixed").reshape(-1).tolist()
            for backend in ACCELERATED
        ]
        self.assertEqual(values[0], values[1])


class BaselineIndependenceTests(unittest.TestCase):
    """A baseline knows nothing about the package under test."""

    def test_no_baseline_module_imports_tensors(self):
        import ast
        import pathlib

        for path in sorted(pathlib.Path("benchmarks/baselines").glob("*.py")):
            with self.subTest(module=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        imported.add(node.module or "")
                    elif isinstance(node, ast.Import):
                        imported.update(alias.name for alias in node.names)
                offenders = {name for name in imported if name.startswith("tensors")}
                self.assertEqual(offenders, set())


if __name__ == "__main__":
    unittest.main()
