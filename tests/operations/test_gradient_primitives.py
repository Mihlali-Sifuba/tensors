"""``sum_to_shape`` as shared gradient-shaping logic, not an operation.

It reverses a broadcast: a value that fed several output positions is owed the
sum of them. Expectations come from that rule and from the broadcasting rules
that decide which axes were stretched.

This module also guards the boundary the refactor drew — that the shaping
function composes existing operations rather than being one, and that the
components removed with the old second derivative method have not returned.
"""

import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state
from tensors.operations.gradient_primitives import sum_to_shape

BACKENDS = ("python", "numpy", "cuda")


class SumToShapeTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def test_a_gradient_already_at_the_target_shape_is_returned_unchanged(self):
        """No reduction is required, so none is performed.

        The same object comes back, which is what lets a same-shape ``a + b``
        hand its gradient straight through with no computation at all.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    gradient = ts.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ts.float64)
                    produced = sum_to_shape(gradient, (2, 2))
                self.assertIs(produced, gradient)

    def test_a_stretched_axis_is_summed_away(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    gradient = ts.Tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64
                    )
                    down = sum_to_shape(gradient, (1, 3))
                    across = sum_to_shape(gradient, (2, 1))
                    everything = sum_to_shape(gradient, (1, 1))
                self.assertEqual(down.tolist(), [5.0, 7.0, 9.0])
                self.assertEqual(across.tolist(), [6.0, 15.0])
                self.assertEqual(everything.tolist(), [21.0])
                for produced, shape in ((down, (1, 3)), (across, (2, 1))):
                    self.assertEqual(tuple(produced.shape), shape)
                    self.assertIs(produced.dtype, ts.float64)
                    self.assertEqual(produced.backend_storage.kind, backend)

    def test_an_axis_the_operand_never_had_is_summed_away(self):
        """Shapes align from the right, so a missing axis counts as stretched."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    gradient = ts.Tensor(
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64
                    )
                    produced = sum_to_shape(gradient, (3,))
                self.assertEqual(produced.tolist(), [5.0, 7.0, 9.0])
                self.assertEqual(tuple(produced.shape), (3,))
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_it_records_when_given_a_graph_value(self):
        """The same statements calculate over Tensors and record over Variables."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                reset_graph_state()
                with ts.use_backend(backend):
                    value = ts.Variable(
                        ts.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=ts.float64)
                    )
                    recorded = sum_to_shape(value, (3,))
                    self.assertIsInstance(recorded, ts.Variable)
                    self.assertEqual(recorded.data.tolist(), [5.0, 7.0, 9.0])
                    # and it differentiates, through the operations it composes
                    back = ts.grad(
                        recorded,
                        value,
                        grad_outputs=ts.Tensor([1.0, 1.0, 1.0], dtype=ts.float64),
                    )
                self.assertEqual(back.tolist(), [1.0] * 6)
                self.assertEqual(tuple(back.shape), (2, 3))
                self.assertEqual(back.backend_storage.kind, backend)

    def test_a_target_that_did_not_broadcast_here_is_refused(self):
        with ts.use_backend("python"):
            gradient = ts.Tensor([[1.0, 2.0, 3.0]], dtype=ts.float64)
            with self.assertRaisesRegex(ValueError, "did not broadcast"):
                sum_to_shape(gradient, (1, 3, 4))
            with self.assertRaisesRegex(ValueError, "did not broadcast"):
                sum_to_shape(gradient, (1, 2))


class GradientPrimitiveBoundaryTests(unittest.TestCase):
    """What the refactor separated, and what it removed."""

    def test_the_shaping_function_is_not_an_operation(self):
        """It composes ``sum`` and ``reshape``, which have derivative rules."""
        import inspect

        from tensors.ops import Operation

        self.assertFalse(isinstance(sum_to_shape, type))
        self.assertFalse(
            any(
                isinstance(member, type) and issubclass(member, Operation)
                for _, member in inspect.getmembers(inspect.getmodule(sum_to_shape))
            ),
            "gradient_primitives holds shaping logic, not operations",
        )
        source = inspect.getsource(sum_to_shape)
        self.assertIn("Sum(", source)
        self.assertIn("reshape(", source)

    def test_the_fused_product_reduction_is_not_here(self):
        """It is a mathematical operation and lives with the reductions."""
        import tensors.operations.gradient_primitives as module

        self.assertFalse(hasattr(module, "ProductSumToShape"))
        from tensors.operations.reductions.product_sum_to_shape import (
            ProductSumToShape,
        )

        self.assertEqual(
            ProductSumToShape.__module__,
            "tensors.operations.reductions.product_sum_to_shape",
        )

    def test_the_removed_module_and_components_have_not_returned(self):
        """``vjp.py`` and the four orphans left with the old graph method."""
        import importlib

        for path in ("tensors.operations.vjp", "tensors.operations._gradient_shaping"):
            with self.subTest(module=path):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(path)

        import tensors.operations.gradient_primitives as primitives
        import tensors.operations.reductions.product_sum_to_shape as reduction

        for name in (
            "ZeroLike",
            "zero_like_graph",
            "MaskedValue",
            "masked_value_graph",
        ):
            with self.subTest(component=name):
                self.assertFalse(hasattr(primitives, name))
                self.assertFalse(hasattr(reduction, name))


if __name__ == "__main__":
    unittest.main()
