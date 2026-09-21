"""Arithmetic lowers its operands inline, not through a helper layer.

The array backends used to route arithmetic operands through an
``_arithmetic_operand`` helper in their ``conversion`` modules. That
indirection is gone: each arithmetic dispatcher now lowers in its own
``selected ==`` branch, so the boundary between a Tensor and a native array is
readable in the one place it happens.

``tensor_to_logical_array`` is deliberately still there. It is not an
arithmetic helper — it is the Tensor-to-native-array boundary that around a
hundred NumPy kernels and thirty CUDA kernels use — so these tests pin that
arithmetic no longer depends on it while everything else still may.
"""

import inspect
import unittest

import tensors as ts
from tensors.backend.dispatch.arithmetic import add as add_dispatch
from tensors.backend.dispatch.arithmetic import divide as divide_dispatch
from tensors.backend.dispatch.arithmetic import multiply as multiply_dispatch
from tensors.backend.dispatch.arithmetic import power as power_dispatch
from tensors.backend.dispatch.arithmetic import subtract as subtract_dispatch
from tests.backend._support import BackendTestCase, requires_cuda, requires_numpy

DISPATCHERS = (
    ("add", add_dispatch),
    ("subtract", subtract_dispatch),
    ("multiply", multiply_dispatch),
    ("divide", divide_dispatch),
    ("power", power_dispatch),
)
ARRAY_BACKENDS = ("numpy", "cuda")


class HelperRemovalTests(unittest.TestCase):
    """The operand helper is gone; the general view boundary is not."""

    def test_no_backend_still_defines_the_operand_helper(self):
        for backend in ARRAY_BACKENDS:
            with self.subTest(backend=backend):
                module = __import__(
                    f"tensors.backend.{backend}.conversion", fromlist=["conversion"]
                )
                self.assertFalse(hasattr(module, "_arithmetic_operand"))

    def test_the_general_view_boundary_is_retained(self):
        """``tensor_to_logical_array`` serves every kernel family, so it is
        not arithmetic's to remove.
        """
        for backend in ARRAY_BACKENDS:
            with self.subTest(backend=backend):
                module = __import__(
                    f"tensors.backend.{backend}.conversion", fromlist=["conversion"]
                )
                self.assertTrue(hasattr(module, "tensor_to_logical_array"))
                self.assertTrue(hasattr(module, "_arithmetic_storage"))

    def test_no_dispatcher_imports_an_operand_helper(self):
        for name, module in DISPATCHERS:
            with self.subTest(operation=name):
                source = inspect.getsource(module)
                self.assertNotIn("_arithmetic_operand", source)
                self.assertNotIn("tensor_to_logical_array", source)


class InlineLoweringTests(BackendTestCase):
    """Each dispatcher lowers in its own branch, from storage and shape."""

    def test_each_dispatcher_lowers_inline_for_both_array_backends(self):
        for name, module in DISPATCHERS:
            source = inspect.getsource(module)
            with self.subTest(operation=name):
                self.assertIn('_logical_storage_for("numpy")', source)
                self.assertIn('_logical_storage_for("cuda")', source)
                self.assertIn("numpy.dtype(dtype.name)", source)
                self.assertIn("cupy.dtype(dtype.name)", source)
                self.assertIn(".buffer.reshape(", source)

    @requires_numpy
    def test_arithmetic_no_longer_routes_through_the_view_helper(self):
        """Patching ``tensor_to_logical_array`` must not disturb arithmetic any more."""
        from unittest.mock import patch

        from tensors.backend.numpy import conversion

        with patch.object(
            conversion,
            "tensor_to_logical_array",
            side_effect=AssertionError("arithmetic must lower inline"),
        ):
            with ts.use_backend("numpy"):
                left = ts.Tensor([1.0, 2.0])
                right = ts.Tensor([3.0, 4.0])
                self.assertEqual((left + right).tolist(), [4.0, 6.0])
                self.assertEqual((left * 2.0).tolist(), [2.0, 4.0])

    @requires_cuda
    def test_cuda_arithmetic_also_lowers_without_the_helper(self):
        from unittest.mock import patch

        from tensors.backend.cuda import conversion

        with patch.object(
            conversion,
            "tensor_to_logical_array",
            side_effect=AssertionError("arithmetic must lower inline"),
        ):
            with ts.use_backend("cuda"):
                left = ts.Tensor([1.0, 2.0])
                right = ts.Tensor([3.0, 4.0])
                self.assertEqual((left + right).tolist(), [4.0, 6.0])


class PreservedBehaviourTests(BackendTestCase):
    """Inlining changed no result, dtype, scalar rule or residency guarantee."""

    def test_every_backend_agrees_after_inlining(self):
        cases = (
            ([1.0, 2.0], [3.0, 4.0], (2,)),
            ([[1.0], [2.0]], [10.0, 20.0], (2, 2)),
        )
        for left_values, right_values, shape in cases:
            with ts.use_backend("python"):
                expected = (
                    ts.Tensor(left_values) + ts.Tensor(right_values)
                ).tolist()
            for backend in ts.available_backends():
                with self.subTest(backend=backend, shape=shape):
                    with ts.use_backend(backend):
                        try:
                            produced = (
                                ts.Tensor(left_values) + ts.Tensor(right_values)
                            )
                        except ts.BackendOperationUnsupportedError:
                            continue
                        self.assertEqual(produced.shape, shape)
                        self.assertEqual(produced.tolist(), expected)
                        self.assertEqual(produced._storage.kind, backend)

    def test_a_scalar_is_lowered_to_the_declared_dtype(self):
        """A bare Python number would let the array API widen the result."""
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    try:
                        value = ts.Tensor([1.0, 2.0], dtype=ts.float32)
                        self.assertIs((value + 2).dtype, ts.float32)
                        self.assertIs((value * 2.0).dtype, ts.float32)
                    except ts.BackendOperationUnsupportedError:
                        continue

    def test_dtype_and_scalar_rules_are_unchanged(self):
        integer = ts.Tensor([1, 2], dtype=ts.int32)
        self.assertIs((integer + 3).dtype, ts.int32)
        self.assertIs((integer + ts.Tensor([1], dtype=ts.int64)).dtype, ts.int64)
        self.assertIs((integer / integer).dtype, ts.float64)
        self.assertRaises(TypeError, lambda: integer + 3.5)
        self.assertRaises(TypeError, lambda: integer + True)

    def test_residency_is_still_enforced(self):
        others = [name for name in ts.available_backends() if name != "python"]
        if not others:
            self.skipTest("only the Python backend is installed")
        for backend in others:
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    foreign = ts.Tensor([1.0, 2.0])
                resident = ts.Tensor([1.0, 2.0])
                with self.assertRaises(ts.BackendMismatchError):
                    resident + foreign


if __name__ == "__main__":
    unittest.main()
