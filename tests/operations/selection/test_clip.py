"""Clamping to a lower bound, an upper bound, or both."""

import unittest

import tensors as ts


class ClipTests(unittest.TestCase):
    def test_clip_supports_two_sided_and_one_sided_bounds(self):
        value = ts.Tensor([-2.0, 0.5, 3.0])

        self.assertEqual(ts.clip(value, 0.0, 1.0).tolist(), [0.0, 0.5, 1.0])
        self.assertEqual(ts.clip(value, min_value=0.0).tolist(), [0.0, 0.5, 3.0])
        self.assertEqual(ts.clip(value, max_value=1.0).tolist(), [-2.0, 0.5, 1.0])

    def test_clip_validates_bounds(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            ts.clip([1.0])
        with self.assertRaisesRegex(ValueError, "greater"):
            ts.clip([1.0], 2.0, 1.0)
        with self.assertRaisesRegex(TypeError, "number"):
            ts.clip([1.0], "zero", 1.0)

    def test_clip_gradient_uses_zero_subgradient_at_boundaries(self):
        value = ts.Variable([-1.0, 0.0, 0.5, 1.0, 2.0])

        first = ts.grad(
            ts.sum(ts.clip(value, 0.0, 1.0)),
            value,
            create_graph=True,
        )
        second = ts.grad(ts.sum(first), value)

        self.assertEqual(first.data.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(second.tolist(), [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
