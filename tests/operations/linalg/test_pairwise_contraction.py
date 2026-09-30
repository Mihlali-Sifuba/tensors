"""Contractions follow the specified products-then-pairwise-sum contract.

Every expectation here is computed by the independent oracle in
``tests/_pairwise_oracle.py`` from docs/summation-semantics.md: each product is
rounded to the dtype and the products of one output element are reduced with
the pairwise tree. The inputs are the hostile ones that used to be refused by
the accelerated backends; now every backend computes them, and all backends
agree bit for bit.
"""

import math
import unittest

import tensors as ts
from tensors.graph.state import reset_graph_state
from tests._pairwise_oracle import left_to_right_sum, pairwise_dot


class PairwiseContractionTests(unittest.TestCase):
    def setUp(self):
        self.previous_backend = ts.get_backend()
        reset_graph_state()

    def tearDown(self):
        ts.set_backend(self.previous_backend)
        reset_graph_state()

    def assertOnEveryBackend(self, operation, expected):
        """Run ``operation`` on every backend; compare to the oracle value."""
        for backend in ts.available_backends():
            with self.subTest(backend=backend):
                reset_graph_state()
                with ts.use_backend(backend):
                    result = operation()
                self.assertEqual(result.backend_storage.kind, backend)
                values = result.tolist()
                for produced, wanted in zip(values, expected, strict=True):
                    if math.isnan(wanted):
                        self.assertTrue(math.isnan(produced))
                    else:
                        self.assertEqual(produced, wanted)

    def test_hostile_finite_cancellation_follows_the_pairwise_tree(self):
        """``[1e16, 1, -1e16, 1]`` pairs as ``(1e16 + 1) + (-1e16 + 1)``.

        Each first-round sum rounds back to ``+-1e16`` (the spacing there is
        2, and the halfway case rounds to even), so the tree gives 0 where
        left-to-right summation gives 1 and the exact sum is 2. The test
        therefore distinguishes the specified tree from both.
        """
        left = [1e16, 1.0, -1e16, 1.0]
        right = [1.0, 1.0, 1.0, 1.0]
        expected = pairwise_dot(left, right)
        self.assertEqual(expected, 0.0)
        self.assertEqual(left_to_right_sum(left), 1.0)
        self.assertEqual(math.fsum(left), 2.0)
        self.assertOnEveryBackend(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)), [expected]
        )

    def test_individual_product_overflow_is_a_rounded_product(self):
        """``fl(1e308 * 1e308)`` and its negation are infinities: NaN."""
        left = [1e308, -1e308]
        right = [1e308, 1e308]
        expected = pairwise_dot(left, right)
        self.assertTrue(math.isnan(expected))
        self.assertOnEveryBackend(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)), [expected]
        )

    def test_each_output_element_is_its_own_tree(self):
        left = [[1.0, 2.0, 3.0, 4.0], [1e16, 1.0, -1e16, 1.0]]
        right = [[1.0], [1.0], [1.0], [1.0]]
        expected = [pairwise_dot(row, [1.0] * 4) for row in left]
        self.assertEqual(expected, [10.0, 0.0])
        self.assertOnEveryBackend(
            lambda: ts.matmul(ts.Tensor(left), ts.Tensor(right)), expected
        )

    def test_a_nonfinite_batch_does_not_affect_a_finite_batch(self):
        left = [math.inf, 0.0, 0.0, 0.0, 1e16, 1.0, -1e16, 1.0]
        expected = [pairwise_dot(left[:4], [1.0] * 4), pairwise_dot(left[4:], [1.0] * 4)]
        self.assertEqual(expected, [math.inf, 0.0])
        self.assertOnEveryBackend(
            lambda: ts.matmul(
                ts.Tensor(left, shape=(2, 1, 4)),
                ts.Tensor([1.0] * 8, shape=(2, 4, 1)),
            ),
            expected,
        )

    def test_hostile_matmul_left_vjp_follows_the_pairwise_tree(self):
        """d(left @ right)/d left sums ``upstream * right`` over the columns."""
        upstream = [1e16, 1.0, -1e16, 1.0]
        expected = pairwise_dot(upstream, [1.0] * 4)

        def operation():
            left = ts.Variable(ts.Tensor([[1.0]]))
            right = ts.Tensor([[1.0, 1.0, 1.0, 1.0]])
            return ts.grad(
                ts.matmul(left, right), left, grad_outputs=ts.Tensor([upstream])
            )

        self.assertOnEveryBackend(operation, [expected])

    def test_hostile_matmul_right_vjp_follows_the_pairwise_tree(self):
        """d(left @ right)/d right sums ``left * upstream`` over the rows."""
        upstream = [1e16, 1.0, -1e16, 1.0]
        expected = pairwise_dot([1.0] * 4, upstream)

        def operation():
            left = ts.Tensor([[1.0], [1.0], [1.0], [1.0]])
            right = ts.Variable(ts.Tensor([[1.0]]))
            return ts.grad(
                ts.matmul(left, right),
                right,
                grad_outputs=ts.Tensor([[value] for value in upstream]),
            )

        self.assertOnEveryBackend(operation, [expected])

    def test_hostile_outer_left_vjp_follows_the_pairwise_tree(self):
        upstream = [1e16, 1.0, -1e16, 1.0]
        expected = pairwise_dot(upstream, [1.0] * 4)

        def operation():
            left = ts.Variable(ts.Tensor([1.0]))
            right = ts.Tensor([1.0, 1.0, 1.0, 1.0])
            return ts.grad(
                ts.outer(left, right), left, grad_outputs=ts.Tensor([upstream])
            )

        self.assertOnEveryBackend(operation, [expected])

    def test_hostile_outer_right_vjp_follows_the_pairwise_tree(self):
        upstream = [1e16, 1.0, -1e16, 1.0]
        expected = pairwise_dot([1.0] * 4, upstream)

        def operation():
            left = ts.Tensor([1.0, 1.0, 1.0, 1.0])
            right = ts.Variable(ts.Tensor([1.0]))
            return ts.grad(
                ts.outer(left, right),
                right,
                grad_outputs=ts.Tensor([[value] for value in upstream]),
            )

        self.assertOnEveryBackend(operation, [expected])


if __name__ == "__main__":
    unittest.main()
