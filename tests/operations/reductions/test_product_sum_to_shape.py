"""``ProductSumToShape`` as a mathematical operation in its own right.

`F(A, B; S) = ReduceToShape(A ⊙ B, S)`: an elementwise product over the
operands' broadcast shape, reduced to a target shape. The expectations here
come from that definition and from the range rules the fused form exists to
respect, never from another backend's output.

The multiplication VJP's use of it is covered in
``tests/autograd/test_backward.py``; these cases are about the operation.
"""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state
from tensors.operations.reductions.product_sum_to_shape import ProductSumToShape

BACKENDS = ("python", "numpy", "cuda")


class ProductSumToShapeTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def _require(self, backend):
        if backend not in ts.available_backends():
            self.skipTest(f"the {backend} backend is not available here")

    def test_without_reduction_it_is_an_elementwise_product(self):
        """The target shape is the operands' own, so nothing is summed."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([1.5, -2.0, 3.0], dtype=ts.float64)
                    right = ts.Tensor([2.0, 4.0, -0.5], dtype=ts.float64)
                    produced = ProductSumToShape(target_shape=(3,)).forward(left, right)
                self.assertEqual(produced.tolist(), [3.0, -8.0, -1.5])
                self.assertEqual(tuple(produced.shape), (3,))
                self.assertIs(produced.dtype, ts.float64)
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_with_reduction_it_sums_the_product_to_the_target_shape(self):
        """``(2, 3) ⊙ (2, 3)`` reduced to ``(1, 3)`` sums down the rows."""
        rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        factors = [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        expected = [
            rows[0][column] * 2.0 + rows[1][column] * 3.0 for column in range(3)
        ]
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor(rows, dtype=ts.float64)
                    right = ts.Tensor(factors, dtype=ts.float64)
                    produced = ProductSumToShape(target_shape=(1, 3)).forward(
                        left, right
                    )
                self.assertEqual(produced.tolist(), expected)
                self.assertEqual(tuple(produced.shape), (1, 3))
                self.assertEqual(produced.backend_storage.kind, backend)

    def test_it_broadcasts_its_operands_before_reducing(self):
        """``(1, 3) ⊙ (2, 1)`` reduces to either operand's shape."""
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1.0, 2.0, 3.0]], dtype=ts.float64)
                    right = ts.Tensor([[10.0], [20.0]], dtype=ts.float64)
                    to_left = ProductSumToShape(target_shape=(1, 3)).forward(
                        left, right
                    )
                    to_right = ProductSumToShape(target_shape=(2, 1)).forward(
                        left, right
                    )
                # column j: 1*a_j*10 + 1*a_j*20 = 30*a_j
                self.assertEqual(to_left.tolist(), [30.0, 60.0, 90.0])
                # row i: b_i * (1 + 2 + 3)
                self.assertEqual(to_right.tolist(), [60.0, 120.0])
                self.assertEqual(tuple(to_left.shape), (1, 3))
                self.assertEqual(tuple(to_right.shape), (2, 1))

    def test_the_fusion_survives_products_that_leave_float_range(self):
        """This is why it is one step and not two.

        ``1 * 1e308`` and ``1 * -1e308`` are each representable, but a naive
        sum of them after rounding is ``inf + -inf``. Grouping the factors
        before rounding keeps the exact reduced result.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1.0], [1.0]], dtype=ts.float64)
                    right = ts.Tensor([[1e308, -1e308]], dtype=ts.float64)
                    fused = ProductSumToShape(target_shape=(2, 1)).forward(left, right)
                self.assertEqual(fused.tolist(), [0.0, 0.0])
                self.assertFalse(any(math.isnan(v) for v in fused.tolist()))
                self.assertEqual(fused.backend_storage.kind, backend)

    def test_products_below_float_range_reduce_to_zero(self):
        """Underflow has no value to recover, and none is invented.

        ``1e-200 * 1e-200`` is ``1e-400``, and the exact sum of two of them,
        ``2e-400``, is below the smallest subnormal. Zero is therefore the
        correctly rounded result; the point is that the grouped form returns
        it rather than a NaN or a spurious magnitude from the logarithms it
        uses when a product does leave the range.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1e-200], [1e-200]], dtype=ts.float64)
                    right = ts.Tensor([[1e-200, 1e-200]], dtype=ts.float64)
                    fused = ProductSumToShape(target_shape=(2, 1)).forward(left, right)
                self.assertEqual(fused.tolist(), [0.0, 0.0])

    def test_a_product_that_overflows_alone_survives_a_finite_reduction(self):
        """``1e300 * 1e300`` is not representable; the reduced result is.

        Two terms of ``+1e300 * 1e300`` and ``-1e300 * 1e300`` cancel to an
        exact zero, which a form that rounded each product first could not
        reach: it would have ``inf`` and ``-inf`` to add.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1e300, -1e300]], dtype=ts.float64)
                    right = ts.Tensor([[1e300, 1e300]], dtype=ts.float64)
                    fused = ProductSumToShape(target_shape=(1, 1)).forward(left, right)
                produced = fused.tolist()[0]
                self.assertFalse(math.isnan(produced))
                self.assertEqual(produced, 0.0)

    def test_the_target_shape_is_configuration_not_forward_state(self):
        """It is recorded with the operation, so a replay reduces the same."""
        operation = ProductSumToShape(target_shape=(1, 3))
        operation.forward(
            ts.Tensor([[1.0, 2.0, 3.0]], dtype=ts.float64),
            ts.Tensor([[1.0, 1.0, 1.0]], dtype=ts.float64),
        )
        self.assertEqual(sorted(type(operation).__slots__), ["target_shape"])
        self.assertEqual(operation.target_shape, (1, 3))
        self.assertNotIsInstance(operation.target_shape, ts.Tensor)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            operation.target_shape = (3,)

    def test_it_is_an_operation_with_its_own_derivative(self):
        from tensors.ops import Operation

        self.assertTrue(issubclass(ProductSumToShape, Operation))
        self.assertIn("forward", vars(ProductSumToShape))
        self.assertIn("backward", vars(ProductSumToShape))
        self.assertEqual(ProductSumToShape.name, "product_sum_to_shape")

    def test_it_lives_with_the_reductions(self):
        """Placed by what it computes, not by which caller it has."""
        self.assertEqual(
            ProductSumToShape.__module__,
            "tensors.operations.reductions.product_sum_to_shape",
        )


if __name__ == "__main__":
    unittest.main()
