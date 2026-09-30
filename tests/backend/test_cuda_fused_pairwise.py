"""The fused CUDA product reduction reproduces the specified tree exactly.

Every expectation comes from the independent oracle in
``tests/_pairwise_oracle.py``. The structural cases are chosen so that the
specified pairwise tree, left-to-right summation, a conventional
stride-halving parallel reduction and the exact sum all differ, so matching
any tree other than the specified one fails.
"""

import math
import random
import struct
import unittest
from fractions import Fraction

import tensors as ts
from tests._pairwise_oracle import pairwise_dot, pairwise_sum, round32

requires_cuda = unittest.skipUnless(
    "cuda" in ts.available_backends(), "CUDA is not available"
)


def _left_to_right(values, rnd):
    total = values[0]
    for value in values[1:]:
        total = rnd(total + value)
    return total


def _stride_halving(values, rnd):
    level = list(values)
    count = len(level)
    while count > 1:
        half = (count + 1) // 2
        for index in range(count // 2):
            level[index] = rnd(level[index] + level[index + half])
        count = half
    return level[0]


def _bits(value, dtype_name):
    if math.isnan(value):
        return b"nan"
    return struct.pack("<f" if dtype_name == "float32" else "<d", value)


@requires_cuda
class FusedPairwiseTests(unittest.TestCase):
    def setUp(self):
        import cupy

        from tensors.backend.cuda.kernels.reductions.fused_pairwise import (
            CHUNK,
            pairwise_indexed_product_sum,
        )

        self.cupy = cupy
        self.chunk = CHUNK
        self.fused = pairwise_indexed_product_sum

    def run_rows(self, left_rows, right_rows, dtype_name):
        """Reduce rows of factor pairs through the fused kernel."""
        cupy = self.cupy
        width = len(left_rows[0])
        left = cupy.asarray([v for row in left_rows for v in row], dtype=dtype_name)
        right = cupy.asarray([v for row in right_rows for v in row], dtype=dtype_name)
        indices = cupy.arange(len(left_rows) * width, dtype=cupy.int64).reshape(-1, width)
        result = self.fused(left, right, indices, indices)
        self.assertIsInstance(result, cupy.ndarray)
        return cupy.asnumpy(result).tolist()

    def assertTree(self, left_rows, right_rows, dtype_name):
        produced = self.run_rows(left_rows, right_rows, dtype_name)
        expected = [pairwise_dot(l, r, dtype_name) for l, r in zip(left_rows, right_rows)]
        self.assertEqual(
            [_bits(v, dtype_name) for v in produced],
            [_bits(v, dtype_name) for v in expected],
        )
        return produced

    def test_the_specified_tree_is_distinguished_from_other_orders(self):
        cases = (
            ("float64", [1.0, 3.0, -2.0, 1e16, 7.0, -1e16, 2.5], 12.0),
            ("float32", [30000000.0, 1.0, -2.0, 3.0, -2.0, -30000000.0, 2.5], 0.0),
        )
        for dtype_name, values, tree_value in cases:
            with self.subTest(dtype=dtype_name):
                rnd = round32 if dtype_name == "float32" else (lambda v: v)
                exact = rnd(float(sum((Fraction(v) for v in values), Fraction(0))))
                others = {
                    _left_to_right(values, rnd),
                    _stride_halving(values, rnd),
                    exact,
                }
                self.assertEqual(pairwise_sum(values, dtype_name), tree_value)
                self.assertNotIn(tree_value, others)
                self.assertEqual(len(others), 3)
                produced = self.assertTree([values], [[1.0] * len(values)], dtype_name)
                self.assertEqual(produced, [tree_value])

    def test_rows_wider_than_one_block_keep_the_global_tree(self):
        pools = {
            "float64": [1e16, -1e16, 1.0, 3.0, -2.0, 2.5, 7.0],
            "float32": [round32(v) for v in (3e7, -3e7, 1.0, 3.0, -2.0, 2.5, 0.75)],
        }
        widths = (
            self.chunk - 1,
            self.chunk,
            self.chunk + 1,
            self.chunk + 6,
            2 * self.chunk + 1,
            25 * self.chunk + 512,
        )
        for dtype_name, pool in pools.items():
            rnd = round32 if dtype_name == "float32" else (lambda v: v)
            for width in widths:
                with self.subTest(dtype=dtype_name, width=width):
                    rng = random.Random(width)
                    values = [rng.choice(pool) for _ in range(width)]
                    if width == self.chunk + 6:
                        # Draw until the four orders all disagree, so this
                        # chunk-spanning case discriminates the tree too.
                        while len({
                            pairwise_sum(values, dtype_name),
                            _left_to_right(values, rnd),
                            _stride_halving(values, rnd),
                            rnd(float(sum((Fraction(v) for v in values), Fraction(0)))),
                        }) < 4:
                            values = [rng.choice(pool) for _ in range(width)]
                    self.assertTree([values], [[1.0] * width], dtype_name)

    def test_nonfinite_products_are_classified_and_zero_is_canonical(self):
        for dtype_name in ("float32", "float64"):
            tiny = struct.unpack("<f", struct.pack("<I", 1))[0] if dtype_name == "float32" else 5e-324
            big = 3e38 if dtype_name == "float32" else 1.7e308
            cases = [
                ([0.0, 1.0], [math.inf, 2.0]),            # 0 * inf -> NaN
                ([math.inf, 1.0, 2.0], [2.0, 3.0, 4.0]),  # inf * finite -> +inf
                ([-math.inf, 1.0], [2.0, 3.0]),           # -inf
                ([math.inf, -math.inf], [1.0, 1.0]),      # both infinities -> NaN
                ([big, -big, 1.0], [big, big, 1.0]),      # overflowed products: +inf, -inf -> NaN
                ([big, 1.0, 1.0], [-big, 1.0, 1.0]),      # overflow to -inf only
                ([math.nan, 1.0], [1.0, 1.0]),            # NaN factor
                ([tiny, tiny, tiny], [1.0, 1.0, 1.0]),    # subnormal products kept
                ([1e-20, 1.0], [1e-20, 0.0]),             # a product that is subnormal
                ([-0.0, -0.0], [1.0, 1.0]),               # -0.0 + -0.0 -> +0.0
                ([1.0, -1.0], [1.0, 1.0]),                # exact cancellation -> +0.0
                ([-0.0], [1.0]),                          # a single -0.0 -> +0.0
            ]
            for left, right in cases:
                with self.subTest(dtype=dtype_name, left=left, right=right):
                    if dtype_name == "float32":
                        left = [round32(v) for v in left]
                        right = [round32(v) for v in right]
                    produced = self.assertTree([left], [right], dtype_name)
                    if produced[0] == 0.0:
                        self.assertEqual(math.copysign(1.0, produced[0]), 1.0)

    def test_random_rows_across_chunk_boundaries(self):
        rng = random.Random(11)
        for dtype_name in ("float32", "float64"):
            rnd = round32 if dtype_name == "float32" else (lambda v: v)
            for width in (1, 2, 3, 150, 400, self.chunk + 3, 3200):
                rows = 6
                left = [[rnd(rng.uniform(-1, 1) * 10.0 ** rng.randint(-8, 8)) for _ in range(width)] for _ in range(rows)]
                right = [[rnd(rng.uniform(-1, 1)) for _ in range(width)] for _ in range(rows)]
                with self.subTest(dtype=dtype_name, width=width):
                    self.assertTree(left, right, dtype_name)

    def test_the_result_stays_on_the_device(self):
        cupy = self.cupy
        values = cupy.arange(8, dtype=cupy.float64)
        indices = cupy.arange(8, dtype=cupy.int64).reshape(2, 4)
        result = self.fused(values, values, indices, indices)
        self.assertIsInstance(result, cupy.ndarray)
        self.assertEqual(result.dtype, cupy.float64)
        self.assertEqual(cupy.asnumpy(result).tolist(), [pairwise_dot([0, 1, 2, 3], [0, 1, 2, 3]), pairwise_dot([4, 5, 6, 7], [4, 5, 6, 7])])


if __name__ == "__main__":
    unittest.main()
