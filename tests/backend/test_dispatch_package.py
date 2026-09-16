from tensors.backend.python.storage import PythonStorage
import ast
import importlib
import importlib.util
import pathlib
import subprocess
import sys
import types
import unittest
import tensors as ts
import tensors.backend as backend_package
from tensors.backend import dispatch as dispatch_package
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.numpy.storage import NumPyStorage
from tensors.backend.storage import Storage
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy

REPRESENTATIVES = {
    "execute_add": "arithmetic",
    "execute_exp_gradient": "elementwise",
    "execute_power_base_gradient": "elementwise",
    "execute_full": "creation",
    "execute_one_hot_targets": "creation",
    "execute_transpose": "manipulation",
    "execute_cast": "manipulation",
    "execute_reduce_sum": "reductions",
    "execute_logsumexp": "reductions",
    "execute_sum_to_shape": "reductions",
    "execute_matmul": "linalg",
    "execute_outer_gradient": "linalg",
    "execute_convolution": "convolution",
    "execute_convolution_gradient": "convolution",
    "execute_fused_elementwise": "fusion",
    "execute_fused_elementwise_backward": "fusion",
    "execute_softmax": "nn",
    "execute_cross_entropy": "nn",
    "execute_validate_distributions": "nn",
    "execute_adam_updates": "optim",
    "execute_rmsprop_update": "optim",
}


class DispatchPackageLayoutTests(unittest.TestCase):
    """Kernel dispatch is grouped by execution domain, not one flat module."""

    MODULES = (
        "arithmetic",
        "elementwise",
        "creation",
        "manipulation",
        "reductions",
        "linalg",
        "convolution",
        "fusion",
        "nn",
        "optim",
    )

    def test_dispatch_is_a_package(self):
        self.assertTrue(hasattr(dispatch_package, "__path__"))
        for name in self.MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(f"tensors.backend.dispatch.{name}")
                self.assertEqual(module.__name__, f"tensors.backend.dispatch.{name}")

    def test_entry_points_belong_to_their_domain_module(self):
        for name, domain in REPRESENTATIVES.items():
            with self.subTest(name=name):
                function = getattr(dispatch_package, name)
                self.assertEqual(
                    function.__module__,
                    f"tensors.backend.dispatch.{domain}.{name.removeprefix('execute_')}",
                )

    def test_every_entry_point_lives_in_a_domain_module(self):
        for name in dispatch_package.__all__:
            function = getattr(dispatch_package, name)
            module = importlib.import_module(function.__module__)
            definitions = [
                node.name
                for node in ast.parse(pathlib.Path(module.__file__).read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith("execute_")
            ]
            self.assertEqual(definitions, [name])

    def test_python_fused_interpretation_stays_internal(self):
        from tensors.backend.python.kernels.fusion.fused_elementwise import (
            fused_elementwise,
        )

        self.assertEqual(
            fused_elementwise.__module__,
            "tensors.backend.python.kernels.fusion.fused_elementwise",
        )
        self.assertFalse(hasattr(backend_package, "_execute_python_fused_elementwise"))

    def test_no_submodule_shares_a_name_with_an_entry_point(self):
        exported = set(dispatch_package.__all__)
        for name in self.MODULES:
            with self.subTest(module=name):
                self.assertNotIn(name, exported)
                self.assertIsInstance(getattr(dispatch_package, name), types.ModuleType)

    def test_domain_modules_do_not_import_one_another(self):
        for name in self.MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(f"tensors.backend.dispatch.{name}")
                tree = ast.parse(pathlib.Path(module.__file__).read_text())
                siblings = sorted(
                    {
                        node.module
                        for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom)
                        and node.level == 1
                        and (node.module in self.MODULES)
                    }
                )
                self.assertEqual(siblings, [])

    def test_every_module_imports_first_without_a_cycle(self):
        for name in ("",) + self.MODULES:
            dotted = "tensors.backend.dispatch" + (f".{name}" if name else "")
            with self.subTest(module=dotted):
                result = subprocess.run(
                    [sys.executable, "-c", f"import {dotted}"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class DispatchFacadeTests(unittest.TestCase):
    """The split is invisible to everything that imports dispatch."""

    def test_backend_facade_still_exposes_every_entry_point(self):
        for name in dispatch_package.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(backend_package, name), getattr(dispatch_package, name)
                )

    def test_documented_import_paths_still_resolve(self):
        from tensors.backend import execute_add, execute_matmul
        from tensors.backend.dispatch import execute_add as package_add

        self.assertIs(execute_add, package_add)
        self.assertIs(execute_matmul, backend_package.execute_matmul)

    def test_entry_points_are_the_domain_functions_not_wrappers(self):
        for name in dispatch_package.__all__:
            with self.subTest(name=name):
                function = getattr(dispatch_package, name)
                owner = importlib.import_module(function.__module__)
                self.assertIs(getattr(owner, name), function)

    def test_dispatch_surface_has_unique_entry_points(self):
        self.assertEqual(
            len(dispatch_package.__all__), len(set(dispatch_package.__all__))
        )
        self.assertTrue(
            all((name.startswith("execute_") for name in dispatch_package.__all__))
        )


class DispatchFallbackTests(BackendTestCase):
    """``None`` still means the caller should run its Python fallback."""

    def test_python_backend_declines_every_domain(self):
        value = ts.Tensor([float(index) for index in range(4_096)])
        matrix = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
        with ts.use_backend("python"):
            self.assertIsInstance(
                dispatch_package.execute_add(
                    value, value, dtype=ts.float64, output_shape=value.shape
                ),
                PythonStorage,
            )
            self.assertIsInstance(
                dispatch_package.execute_exp(value, dtype=ts.float64), PythonStorage
            )
            self.assertIsInstance(
                dispatch_package.execute_full((4_096,), 1.0, dtype=ts.float64),
                PythonStorage,
            )
            self.assertIsInstance(
                dispatch_package.execute_transpose(matrix, (1, 0), output_shape=(2, 2)),
                PythonStorage,
            )
            self.assertIsInstance(
                dispatch_package.execute_reduce_sum(
                    value, (0,), keepdims=False, dtype=ts.float64, output_shape=()
                ),
                PythonStorage,
            )
            self.assertIsInstance(
                dispatch_package.execute_matmul(
                    matrix, matrix, dtype=ts.float64, output_shape=(2, 2)
                ),
                PythonStorage,
            )

    @requires_numpy
    def test_work_below_the_threshold_still_declines(self):
        tiny = ts.Tensor([1.0, 2.0])
        with ts.use_backend("numpy"):
            self.assertIsInstance(
                dispatch_package.execute_add(
                    tiny, tiny, dtype=ts.float64, output_shape=tiny.shape
                ),
                PythonStorage,
            )
            self.assertIsInstance(
                dispatch_package.execute_full((2,), 1.0, dtype=ts.float64),
                PythonStorage,
            )

    def test_python_backend_interprets_a_fused_chain_itself(self):
        value = ts.Tensor([1.0, 2.0, 3.0, 4.0])
        steps = (("add", 1.0, False, None), ("multiply", 2.0, False, None))
        with ts.use_backend("python"):
            storages = dispatch_package.execute_fused_elementwise(
                (value,), steps, dtype=ts.float64, output_shape=value.shape
            )
        self.assertIsNotNone(storages)
        self.assertEqual(list(storages[-1].buffer), [4.0, 6.0, 8.0, 10.0])


class NumPyDispatchTests(BackendTestCase):
    """Accelerated dispatch reaches the NumPy kernels through the package."""

    LARGE = 4_096

    def _value(self):
        return ts.full((self.LARGE,), 2.0)

    @requires_numpy
    def test_each_domain_returns_native_storage(self):
        value = self._value()
        matrix = ts.full((64, 64), 1.5)
        with ts.use_backend("numpy"):
            results = {
                "arithmetic": dispatch_package.execute_add(
                    value, value, dtype=ts.float64, output_shape=value.shape
                ),
                "creation": dispatch_package.execute_full(
                    (self.LARGE,), 3.0, dtype=ts.float64
                ),
                "manipulation": dispatch_package.execute_transpose(
                    matrix, (1, 0), output_shape=(64, 64)
                ),
                "reductions": dispatch_package.execute_reduce_sum(
                    value, (0,), keepdims=False, dtype=ts.float64, output_shape=()
                ),
                "linalg": dispatch_package.execute_matmul(
                    matrix, matrix, dtype=ts.float64, output_shape=(64, 64)
                ),
                "nn": dispatch_package.execute_softmax(value, 0, dtype=ts.float64),
            }
        for domain, storage in results.items():
            with self.subTest(domain=domain):
                self.assertIsInstance(storage, Storage)
                self.assertIsInstance(storage, NumPyStorage)

    @requires_numpy
    def test_dispatch_result_matches_the_python_fallback(self):
        value = self._value()
        with ts.use_backend("numpy"):
            storage = dispatch_package.execute_add(
                value, value, dtype=ts.float64, output_shape=value.shape
            )
        self.assertEqual(list(storage.buffer)[:4], [4.0, 4.0, 4.0, 4.0])
        with ts.use_backend("python"):
            expected = (value + value).tolist()
        self.assertEqual(list(storage.buffer), expected)


@requires_cuda
class CudaDispatchTests(BackendTestCase):
    """The same entry points keep CUDA results on the device."""

    LARGE = 4_096

    def test_each_domain_returns_device_storage(self):
        value = ts.full((self.LARGE,), 2.0)
        matrix = ts.full((64, 64), 1.5)
        with ts.use_backend("cuda"):
            results = {
                "arithmetic": dispatch_package.execute_add(
                    value, value, dtype=ts.float64, output_shape=value.shape
                ),
                "creation": dispatch_package.execute_full(
                    (self.LARGE,), 3.0, dtype=ts.float64
                ),
                "reductions": dispatch_package.execute_reduce_sum(
                    value, (0,), keepdims=False, dtype=ts.float64, output_shape=()
                ),
                "linalg": dispatch_package.execute_matmul(
                    matrix, matrix, dtype=ts.float64, output_shape=(64, 64)
                ),
            }
        for domain, storage in results.items():
            with self.subTest(domain=domain):
                self.assertIsInstance(storage, CudaStorage)

    def test_cuda_dispatch_matches_the_python_backend(self):
        value = ts.full((self.LARGE,), 2.0)
        with ts.use_backend("cuda"):
            storage = dispatch_package.execute_add(
                value, value, dtype=ts.float64, output_shape=value.shape
            )
        with ts.use_backend("python"):
            expected = (value + value).tolist()
        self.assertEqual(storage.copy().buffer.get().tolist(), expected)


if __name__ == "__main__":
    unittest.main()
