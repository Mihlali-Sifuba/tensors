"""``ProductSumToShape`` as a mathematical operation in its own right.

`F(A, B; S) = ReduceToShape(A ⊙ B, S)`: an elementwise product over the
operands' broadcast shape, each product rounded to the dtype, then reduced to
a target shape with the package's pairwise tree. The expectations here come
from that definition (docs/summation-semantics.md), never from another
backend's output.

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

    def test_a_two_term_cancellation_is_one_exact_addition(self):
        """``1 * 1e308`` and ``1 * -1e308`` are both exact products.

        A two-term group is a single rounded addition, and ``1e308 + -1e308``
        is exactly zero.
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
        """``fl(1e-200 * 1e-200)`` underflows to zero, and ``0 + 0`` is zero.

        Each product is rounded to the dtype before the reduction, so the
        group is two zeros; no NaN or spurious magnitude appears.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1e-200], [1e-200]], dtype=ts.float64)
                    right = ts.Tensor([[1e-200, 1e-200]], dtype=ts.float64)
                    fused = ProductSumToShape(target_shape=(2, 1)).forward(left, right)
                self.assertEqual(fused.tolist(), [0.0, 0.0])

    def test_each_product_is_rounded_before_the_reduction(self):
        """``fl(1e300 * 1e300)`` is ``+inf`` and ``fl(-1e300 * 1e300)`` is ``-inf``.

        The products are rounded to the dtype first, so the group holds both
        infinities and the specified result is NaN, on every backend. The
        exact reduced value would be zero; the contract is the rounded
        products then the pairwise tree.
        """
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self._require(backend)
                with ts.use_backend(backend):
                    left = ts.Tensor([[1e300, -1e300]], dtype=ts.float64)
                    right = ts.Tensor([[1e300, 1e300]], dtype=ts.float64)
                    fused = ProductSumToShape(target_shape=(1, 1)).forward(left, right)
                produced = fused.tolist()[0]
                self.assertTrue(math.isnan(produced))
                self.assertEqual(fused.backend_storage.kind, backend)

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
