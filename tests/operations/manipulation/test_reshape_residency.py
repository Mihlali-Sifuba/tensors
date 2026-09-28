"""Reshape moves no values through the host.

`docs/memory-model.md` already described reshape as gathering through
``_logical_storage_for`` and reshaping the compact logical values. The
operation did not: it rebuilt the tensor from ``_data``, which materialises
every value on the host, so reshaping a device tensor cost a
device-to-host-to-device round trip for values it never inspects.
"""

import unittest
from unittest.mock import patch

import tensors as ts
import tensors.tensor as tensor_module


def available_backends():
    return ts.available_backends()


class HostReadCounter:
    """Count materialisations of a Tensor's host values."""

    def __init__(self):
        self.reads = 0
        self._original = tensor_module.Tensor._data.fget

    def __enter__(self):
        counter = self

        def counting(instance):
            counter.reads += 1
            return counter._original(instance)

        self._patch = patch.object(tensor_module.Tensor, "_data", property(counting))
        self._patch.__enter__()
        return self

    def __exit__(self, *arguments):
        return self._patch.__exit__(*arguments)


class ReshapeResidencyTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()

    def tearDown(self):
        ts.set_backend(self.previous_backend)

    def test_reshape_reads_no_host_values(self):
        for backend in available_backends():
            for dtype in (ts.float64, ts.float32):
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.full((1024,), 2.0, dtype=dtype)
                        with HostReadCounter() as counter:
                            ts.reshape(value, (32, 32))
                    self.assertEqual(
                        counter.reads,
                        0,
                        "reshape must not materialise values on the host",
                    )

    def test_reshape_keeps_the_operands_residency(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.full((1024,), 2.0)
                    produced = ts.reshape(value, (32, 32))
                self.assertEqual(produced.backend_storage.kind, backend)
                self.assertEqual(produced.shape, (32, 32))

    def test_a_one_element_reshape_stays_on_the_selected_backend(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    produced = ts.reshape(ts.Tensor([5.0]), (1, 1))
                self.assertEqual(produced.backend_storage.kind, backend)
                self.assertEqual(produced.tolist(), [5.0])

    def test_a_non_contiguous_operand_reshapes_by_logical_order(self):
        """A view must not be reshaped by its flat buffer."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                    transposed = ts.transpose(value)
                    self.assertEqual(transposed.shape, (3, 2))
                    produced = ts.reshape(transposed, (6,))
                # Logical row-major order of the transpose, not the buffer.
                self.assertEqual(produced.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_a_non_contiguous_operand_reads_no_host_values(self):
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.full((32, 32), 3.0)
                    transposed = ts.transpose(value)
                    with HostReadCounter() as counter:
                        ts.reshape(transposed, (1024,))
                self.assertEqual(counter.reads, 0)

    def test_the_result_is_independently_owned(self):
        """Reshape returns a compact tensor that does not alias its source."""
        for backend in available_backends():
            with self.subTest(backend=backend):
                with ts.use_backend(backend):
                    value = ts.Tensor([1.0, 2.0, 3.0, 4.0])
                    produced = ts.reshape(value, (2, 2))
                    self.assertIsNot(produced.backend_storage, value.backend_storage)
                    value[0] = 9.0
                    self.assertEqual(produced.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_dtype_and_values_are_preserved(self):
        for backend in available_backends():
            for dtype in (ts.float64, ts.float32, ts.int64, ts.uint8):
                with self.subTest(backend=backend, dtype=dtype.name):
                    with ts.use_backend(backend):
                        value = ts.Tensor([1, 2, 3, 4], dtype=dtype)
                        produced = ts.reshape(value, (2, 2))
                    self.assertIs(produced.dtype, dtype)
                    self.assertEqual(produced.tolist(), [1, 2, 3, 4])

    def test_a_size_mismatch_still_raises(self):
        with ts.use_backend("python"):
            with self.assertRaisesRegex(ValueError, "Cannot reshape"):
                ts.reshape(ts.Tensor([1.0, 2.0, 3.0]), (2, 2))

    def test_reshape_does_not_name_the_host_accessor(self):
        """Structural: the forward must not reach for ``_data``."""
        import ast
        import importlib
        import inspect
        import textwrap

        module = importlib.import_module("tensors.operations.manipulation.reshape")
        tree = ast.parse(textwrap.dedent(inspect.getsource(module.Reshape.forward)))
        host_reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "_data"
        ]
        self.assertEqual(host_reads, [])


if __name__ == "__main__":
    unittest.main()
